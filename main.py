import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from net.NTSCC_Hyperior import NTSCC_Hyperprior
import torch.optim as optim
from utils import *
from data.datasets import get_loader
import torch
from config import config
import time


def train_one_epoch():
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
                f'Lr {cur_lr}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()


def test():
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
    cbr_sideinfo = np.log2(config.multiple_rate.__len__()) / (16*16*3) / np.log2(1 + 10 ** (net.channel.chan_param / 10))
    logger.info(f'Finish test! Average PSNR={psnr_jsccs.avg:.4f}dB, CBR={cbrs.avg + cbr_sideinfo:.4f}')


if __name__ == '__main__':
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = NTSCC_Hyperprior(config).cuda()
    model_path = config.checkpoint
    load_weights(net, model_path)
    train_loader, test_loader = get_loader(config)

    cur_lr = config.lr
    G_params = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
    aux_params = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
    optimizer_G = optim.Adam(G_params, lr=cur_lr)
    aux_optimizer = optim.Adam(aux_params, lr=1e-3)

    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    test()
    # tot_epoch = 5000
    # for epoch in range(steps_epoch, tot_epoch):
    #     device = next(net.parameters()).device
    #     train_one_epoch()
    #     if (epoch + 1) % 100 == 0:
    #         save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
    #         test()
