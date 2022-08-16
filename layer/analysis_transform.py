import torch.nn as nn
from timm.models.layers import trunc_normal_
from layer.layers import SwinTransformerBlock, PatchEmbed, PatchMerging
import torch


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=out_dim, input_resolution=self.input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for _, blk in enumerate(self.blocks):
            blk.input_resolution = (H // 2, W // 2)
            blk.update_mask()
        if self.downsample is not None:
            self.downsample.input_resolution = (H, W)


class AnalysisTransform(nn.Module):
    def __init__(self, img_size=(256, 256),
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True):
        super().__init__()
        self.num_layers = len(embed_dims)
        self.patch_norm = patch_norm
        self.num_features = embed_dims[-1]
        self.mlp_ratio = mlp_ratio
        self.patches_resolution = img_size
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)
        self.patch_embed = PatchEmbed(img_size, 2, 3, embed_dims[0], nn.LayerNorm)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else embed_dims[0],
                               out_dim=int(embed_dims[i_layer]),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer != 0 else None)
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.patch_embed(x)  # BLN
        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
        x = self.norm(x)
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        # flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        # flops += self.num_features * self.num_classes
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.H = H // (2 ** self.num_layers)
        self.W = W // (2 ** self.num_layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** i_layer),
                                    W // (2 ** i_layer))


def build_model(config):
    input_image = torch.ones([8, 3, 256, 256]).to(config.device)
    model = AnalysisTransform(**config.ga_kwargs).to(config.device)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
    model(input_image)


if __name__ == '__main__':
    import torch
    import sys
    from datetime import datetime

    sys.path.append("/media/D/wangsixian/DJSCC")


    class config:
        seed = 1024
        pass_channel = True
        CUDA = True
        device = torch.device("cuda:0")
        norm = False  # 计算MSE LOSS 的时候是否重新归一化
        base_path = '/home/wangsixian/'
        trainset = 'OpenImages'
        train_data_dir = [base_path + 'Dataset/openimages/**']
        test_data_dir = ['/media/D/Dataset/kodak_test']

        # logger
        print_step = 20
        plot_step = 10000
        filename = datetime.now().__str__()[:-7]
        workdir = './history/{}'.format(filename)
        log = workdir + '/Log_{}.log'.format(filename)
        samples = workdir + '/samples'
        models = workdir + '/models'
        logger = None

        distortion_metric = 'MSE'

        # NTC training details
        image_dims = (3, 256, 256)
        normalize = False
        lr = {
            "base": 0.0001,
            "decay": 0.1,
            "decay_interval": 920000
        }
        train_lambda = 1024
        warmup_step = 1000
        tot_step = 2500000
        tot_epoch = 10000000
        save_model_freq = 40000
        test_step = 500000
        batch_size = 10
        out_channel_N = 192
        out_channel_M = 256
        # out_channel_N = 256
        # out_channel_M = 384

        # DJSCC details
        random_snr = True
        channel = {"type": 'awgn', 'chan_param': 1}
        multiple_snr = [10]

        random_eta = True
        multiple_eta = [0.2]

        num_rates = 16
        multiple_rate = [16, 32, 48, 64, 80, 96, 102, 118, 134, 160, 186, 192, 208, 224, 240, 256]
        if train_lambda >= 128:
            # eta 0.4 train_lambda < 1 / 4
            multiple_rate = [0, 2, 4, 6, 10, 14, 18, 24, 32, 40, 48, 80, 102, 160, 208, 256]

        ga_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        gs_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=128,
            embed_dims=[256, 256, 256, 256], depths=[4, 2, 1, 1], num_heads=[8, 8, 8, 8],
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            # embed_dims=[384, 384, 384, 384], depths=[4, 2, 1, 1], num_heads=[8, 8, 8, 8],
            # window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            SNR_choice=multiple_snr,
            eta_choice=multiple_eta,
            rate_choice=multiple_rate, NTC=True
        )

        jscc_encoder_kwargs = dict(
            patches_resolution=[image_dims[1] // 16, image_dims[2] // 16], in_chans=256,
            embed_dims=[256], depths=[4], num_heads=[8], bottleneck_dim=384,
            window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            SNR_choice=multiple_snr,
            eta_choice=multiple_eta,
            rate_choice=multiple_rate
        )

        jscc_decoder_kwargs = dict(
            patches_resolution=[image_dims[1] // 16, image_dims[2] // 16], in_chans=256,
            embed_dims=[256], depths=[4], num_heads=[8], bottleneck_dim=384,
            window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            SNR_choice=multiple_snr,
            eta_choice=multiple_eta,
            rate_choice=multiple_rate
        )


    from utils import *

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger = logger_configuration(config, save_log=False)
    build_model(config)
