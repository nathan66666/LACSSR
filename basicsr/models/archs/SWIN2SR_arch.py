# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
NAFSSR: Stereo Image Super-Resolution Using NAFNet

@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops.layers.torch import Rearrange
import timm

from basicsr.models.archs.NAFNet_arch import LayerNorm2d, NAFBlock
from basicsr.models.archs.arch_util import MySequential, trunc_normal_
from basicsr.models.archs.local_arch import Local_Base

from einops import rearrange
import math

from basicsr.models.archs.swin2sr_arch import RSTB, UpsampleOneStep, PatchUnEmbed, PatchEmbed, Upsample


class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, feat, x_size):
        b, _, c = feat[0].size()
        x_l, x_r = [x.permute(0, 2, 1).contiguous().view(b, c, x_size[0], x_size[1]) for x in feat]

        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1).contiguous()  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3).contiguous()  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1).contiguous()  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1).contiguous()  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2).contiguous(), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2).contiguous() * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2).contiguous() * self.gamma

        x_l = x_l + F_r2l
        x_r = x_r + F_l2r

        feat = [x_l.flatten(2).permute(0, 2, 1), x_r.flatten(2).permute(0, 2, 1)]
        return feat


class ST2NetSR(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
          
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=4,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 dual=False,
                 fusion=False):
        super().__init__()
        self.dual = dual
        self.fusion = fusion
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
             
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)

        # build fusion
        if self.fusion:
            self.layers_f = nn.ModuleList()
            for i_layer in range(self.num_layers):
                self.layers_f.append(SCAM(embed_dim))

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

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

    def forward_features(self, feat):
        x_size = (feat[0].shape[2], feat[0].shape[3])
        feat = [self.patch_embed(x) for x in feat]
        if self.ape:
            feat = [x + self.absolute_pos_embed for x in feat]
        feat = [self.pos_drop(x) for x in feat]

        for i in range(len(self.layers)):
            feat = [self.layers[i](x, x_size) for x in feat]
            if self.fusion:
                feat = self.layers_f[i](feat, x_size)

        feat = [self.norm(x) for x in feat]
        feat = [self.patch_unembed(x, x_size) for x in feat]

        return feat

    def forward(self, inp):
        # inp_hr = F.interpolate(inp, scale_factor=4, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp,)

        self.mean = self.mean.type_as(inp[0])
        inp = [(x - self.mean) * self.img_range for x in inp]

        if self.upsampler == 'pixelshuffle':
            # for classical SR

            feat = [self.conv_first(x) for x in inp]
            feat_res = self.forward_features(feat)
            feat = [self.conv_after_body(x_res) + x for x, x_res in zip(feat, feat_res)]
            feat = [self.conv_before_upsample(x) for x in feat]
            feat = [self.conv_last(self.upsample(x)) for x in feat]
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            feat = self.conv_first(inp)
            feat = self.conv_after_body(self.forward_features(feat)) + feat
            feat = self.upsample(feat)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            feat = [self.conv_first(x) for x in inp]
            feat_res = self.forward_features(feat)
            feat = [self.conv_after_body(x_res) + x for x, x_res in zip(feat, feat_res)]
            feat = [self.conv_before_upsample(x) for x in feat]

            feat = [self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest'))) for x in feat]
            feat = [self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest'))) for x in feat]
            feat = [self.conv_last(self.lrelu(self.conv_hr(x))) for x in feat]
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(inp)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            feat = inp + self.conv_last(res)

        out = [x / self.img_range + self.mean for x in feat]
        out = torch.cat(out, dim=1)
        # out = out + inp_hr
        return out

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class SWIN2SR(Local_Base, ST2NetSR):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
             
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 train_size=(1, 3, 32, 32),
                 fast_imp=False,
                 dual=False,
                 fusion=False,
                 **kwargs
                 ):
        Local_Base.__init__(self)
        ST2NetSR.__init__(self,
                         img_size=img_size,
                         patch_size=patch_size,
                         in_chans=in_chans,
                         embed_dim=embed_dim,
                         depths=depths,
                         num_heads=num_heads,
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                       
                         drop_rate=drop_rate,
                         attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate,
                         norm_layer=norm_layer,
                         ape=ape,
                         patch_norm=patch_norm,
                         use_checkpoint=use_checkpoint,
                         upscale=upscale,
                         img_range=img_range,
                         upsampler=upsampler,
                         resi_connection=resi_connection,
                         dual=dual,
                         fusion=fusion, )

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    train_size = (1, 6, 32, 32)
    model = STSSR(
        upscale=4,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        train_size=train_size,
        dual=True,
        fusion=False
    )
    print(model)
    # print(height, width, model.flops() / 1e9)
    inp_shape = (6, 64, 64)

    from ptflops import get_model_complexity_info

    FLOPS = 0
    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=True)

    # params = float(params[:-4])
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs, params)

    from basicsr.models.archs.arch_util import measure_inference_speed

    # model = model.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(model, (data,))
