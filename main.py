import os
import time
from datetime import datetime
import sys
import random
import argparse
from net.NTSCC_Hyperior import NTSCC_Hyperprior
import torch.optim as optim
from utils import *
from data.datasets import get_loader, get_test_loader
from config import config


def train_one_epoch(epoch, net, train_loader, optimizer_G, aux_optimizer, device, logger):
    global global_step
    net.train()
    elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs = [AverageMeter() for _ in range(7)]
    metrics = [elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs]
    for batch_idx, input_image in enumerate(train_loader):
        optimizer_G.zero_grad()
        aux_optimizer.zero_grad()

        start_time = time.time()
        input_image = input_image.to(device)
        global_step += 1
        mse_loss_ntc, bpp_y, bpp_z, mse_loss_ntscc, cbr_y, x_hat_ntc, x_hat_ntscc = net(input_image)
        if config.use_side_info:
            cbr_z = bpp_snr_to_kdivn(bpp_z, 10)
            loss = mse_loss_ntscc + mse_loss_ntc + config.train_lambda * (bpp_y * config.eta + cbr_z)
            cbrs.update(cbr_y + cbr_z)
        else:
            # add ntc_loss to improve the training convergence stability
            ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
            loss = ntc_loss + mse_loss_ntscc
            cbrs.update(cbr_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer_G.step()

        aux_loss = net.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        bppys.update(bpp_y.item())
        bppzs.update(bpp_z.item())

        psnr_jscc = 10 * (torch.log(255. * 255. / mse_loss_ntscc) / np.log(10))
        psnr_jsccs.update(psnr_jscc.item())
        psnr = 10 * (torch.log(255. * 255. / mse_loss_ntc) / np.log(10))
        psnrs.update(psnr.item())

        if (global_step % config.print_step) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log = (' | '.join([
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR_JSCC {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'PSNR_NTC {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
                f'Epoch {epoch}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()


def test(net, test_loader, logger):
    with torch.no_grad():
        net.eval()
        elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs = [AverageMeter() for _ in range(7)]
        PSNR_list = []
        CBR_list = []
        for batch_idx, input_image in enumerate(test_loader):
            start_time = time.time()
            input_image = input_image.cuda()
            mse_loss_ntc, bpp_y, bpp_z, mse_loss_ntscc, cbr_y, x_hat_ntc, x_hat_ntscc = net(input_image)
            if config.use_side_info:
                cbr_z = bpp_snr_to_kdivn(bpp_z, 10)
                ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
                ntscc_loss = mse_loss_ntscc + bpp_y * config.eta + cbr_z
                loss = ntc_loss + ntscc_loss
                cbrs.update(cbr_y + cbr_z)
            else:
                ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
                loss = ntc_loss + mse_loss_ntscc
                cbrs.update(cbr_y)
            losses.update(loss.item())
            bppys.update(bpp_y)
            bppzs.update(bpp_z)
            elapsed.update(time.time() - start_time)

            psnr_jscc = CalcuPSNR_int(input_image, x_hat_ntscc).mean()
            psnr_jsccs.update(psnr_jscc)
            psnr = CalcuPSNR_int(input_image, x_hat_ntc).mean()
            psnrs.update(psnr)
            log = (' | '.join([
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.val:.2f}',
                f'PSNR1 {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'PSNR2 {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
            ]))
            logger.info(log)
            PSNR_list.append(psnr_jscc)
            CBR_list.append(cbr_y)

    # Here, the channel bandwidth cost of side info \bar{k} is transmitted by a capacity-achieving channel code. Note
    # that, the side info should be transmitted through entropy coding and channel coding, which will be addressed in
    # future releases.

    # capacity-achieving channel code
    cbr_sideinfo = np.log2(config.multiple_rate.__len__()) / (16 * 16 * 3) / np.log2(
        1 + 10 ** (net.channel.chan_param / 10))

    # 2/3 rate LDPC + 16QAM for AWGN SNR=10dB
    # cbr_sideinfo = np.log2(config.multiple_rate.__len__()) / (16 * 16 * 8)
    logger.info(f'Finish test! Average PSNR={psnr_jsccs.avg:.4f}dB, CBR={cbrs.avg + cbr_sideinfo:.4f}')


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training/testing script.")
    parser.add_argument(
        "-p",
        "--phase",
        default='test',  # train
        type=str,
        help="Train or Test",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5000,
        type=int,
        help="Number of epochs (default: %(default)s)"
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=0,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=1024, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        '--name',
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
        type=str,
        help='Result dir name',
    )
    parser.add_argument(
        '--save_log', action='store_true', default=True, help='Save log to disk'
    )
    parser.add_argument("--checkpoint",
                        default="checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth",
                        type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.device = device

    workdir, logger = logger_configuration(args.name, phase=args.phase, save_log=args.save_log)
    config.logger = logger
    logger.info(config.__dict__)

    net = NTSCC_Hyperprior(config).cuda()
    model_path = args.checkpoint
    load_weights(net, model_path)


    if args.phase == 'test':
        test_loader = get_test_loader(config)
        test(net, test_loader, logger)
    elif args.phase == 'train':
        train_loader, test_loader = get_loader(config)
        global global_step
        G_params = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
        aux_params = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
        optimizer_G = optim.Adam(G_params, lr=config.lr)
        aux_optimizer = optim.Adam(aux_params, lr=config.aux_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[4000, 4500], gamma=0.1)
        tot_epoch = 5000
        global_step = 0
        best_loss = float("inf")
        steps_epoch = global_step // train_loader.__len__()
        for epoch in range(steps_epoch, tot_epoch):
            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer_G.param_groups[0]['lr']}")
            train_one_epoch(epoch, net, train_loader, optimizer_G, aux_optimizer, device, logger)
            lr_scheduler.step()

            loss = test(net, test_loader, logger)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if is_best:
                save_model(net, save_path=workdir + '/models/EP{}_best_loss.model'.format(epoch + 1))
                test(net, test_loader, logger)

            if (epoch + 1) % 100 == 0:
                save_model(net, save_path=workdir + '/models/EP{}.model'.format(epoch + 1))


if __name__ == '__main__':
    main(sys.argv[1:])
