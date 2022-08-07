import torch
import datetime
import torch.nn as nn


class config:
    seed = 1024
    CUDA = True
    device = torch.device("cuda:0")
    base_path = '/media/Dataset/HR_Image_dataset'
    checkpoint = '/media/D/wangsixian/NTSCC_JSAC/checkpoints/update_PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_1_psnr.pth'
    train_data_dir = [base_path + '/DIV2K_train_HR']
    test_data_dir = ['/media/Dataset_/Dataset/kodak_test']
    print_step = 50
    plot_step = 1000
    filename = datetime.datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    distortion_metric = 'MSE'

    # training details
    image_dims = (3, 256, 256)
    lr = 1e-4
    batch_size = 10

    use_side_info = False
    train_lambda = 64
    eta = 0.2

    channel = {"type": 'awgn', 'chan_param': 10}
    multiple_rate = [16, 32, 48, 64, 80, 96, 102, 118, 134, 160, 186, 192, 208, 224, 240, 256]
    ga_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[256, 256, 256, 256], depths=[1, 1, 2, 4], num_heads=[8, 8, 8, 8],
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )

    gs_kwargs = dict(
        img_size=(image_dims[1], image_dims[2]),
        embed_dims=[256, 256, 256, 256], depths=[4, 2, 1, 1], num_heads=[8, 8, 8, 8],
        window_size=8, mlp_ratio=4., norm_layer=nn.LayerNorm, patch_norm=True
    )

    fe_kwargs = dict(
        input_resolution=(image_dims[1] // 16, image_dims[2] // 16),
        embed_dim=256, depths=[4], num_heads=[8],
        window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=multiple_rate
    )

    fd_kwargs = dict(
        input_resolution=(image_dims[1] // 16, image_dims[2] // 16),
        embed_dim=256, depths=[4], num_heads=[8],
        window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, rate_choice=multiple_rate
    )