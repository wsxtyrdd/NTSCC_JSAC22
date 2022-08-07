import math
import torch.nn as nn
import torch
from layer.layers import BasicLayerDec
from timm.models.layers import trunc_normal_
import numpy as np



class RateAdaptionDecoder(nn.Module):
    def __init__(self, channel_num, rate_choice, mode='CHW'):
        super(RateAdaptionDecoder, self).__init__()
        self.C = channel_num
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.weight = nn.Parameter(torch.zeros(self.rate_num, max(self.rate_choice), self.C))
        self.bias = nn.Parameter(torch.zeros(self.rate_num, self.C))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.rate_num)
        torch.nn.init.uniform_(self.bias, -bound, bound)
        # trunc_normal_(self.weight_bias, std=.02)

    def forward(self, x, indexes):
        B, _, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        w = torch.index_select(self.weight, 0, indexes).reshape(B, H * W, max(self.rate_choice), self.C)
        b = torch.index_select(self.bias, 0, indexes).reshape(B, H * W, self.C)
        # print(w.dtype)
        # print(b.dtype)
        # print(x_BLC.dtype)
        x_BLC = torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b  # BLN
        out = x_BLC.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return out


class JSCCDecoder(nn.Module):
    def __init__(self, embed_dim=256, depths=[1, 1, 1],
                 input_resolution=(16, 16),
                 num_heads=[8, 8, 8], window_size=(8, 16, 16),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, rate_choice=[0, 128, 256]):
        super(JSCCDecoder, self).__init__()
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayerDec(dim=embed_dim, out_dim=embed_dim, input_resolution=input_resolution,
                                  depth=depths[i_layer], num_heads=num_heads[i_layer],
                                  window_size=window_size, mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale, upsample=None)
            self.layers.append(layer)
        self.embed_dim = embed_dim
        self.rate_adaption = RateAdaptionDecoder(embed_dim, rate_choice)
        self.rate_choice = rate_choice
        self.rate_num = len(rate_choice)
        self.register_buffer("rate_choice_tensor", torch.tensor(np.asarray(rate_choice)))
        self.rate_token = nn.Parameter(torch.zeros(self.rate_num, embed_dim))
        trunc_normal_(self.rate_token, std=.02)

    def forward(self, x, indexes):
        B, _, H, W = x.size()
        x = self.rate_adaption(x, indexes)

        x_BLC = x.flatten(2).permute(0, 2, 1)
        rate_token = torch.index_select(self.rate_token, 0, indexes)  # BL, N
        rate_token = rate_token.reshape(B, H * W, self.embed_dim)

        x_BLC = x_BLC + rate_token
        for layer in self.layers:
            x_BLC = layer(x_BLC.contiguous())
        x_BCHW = x_BLC.reshape(B, H, W, self.embed_dim).permute(0, 3, 1, 2)
        return x_BCHW

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H, W)
