import torch
import datetime
import torch.nn as nn


class config:

    train_data_dir = ['/media/Dataset/HR_Image_dataset/DIV2K_train_HR']
    test_data_dir = ['/media/Dataset_/Dataset/kodak_test']
    batch_size = 10
    num_workers = 8

    print_step = 50
    plot_step = 1000
    logger = None

    # training details
    image_dims = (3, 256, 256)
    lr = 1e-4
    aux_lr = 1e-3
    distortion_metric = 'MSE'  # 'MS-SSIM'

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