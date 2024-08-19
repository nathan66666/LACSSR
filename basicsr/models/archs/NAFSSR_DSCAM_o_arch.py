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

from basicsr.models.archs.NAFNet_arch import LayerNorm2d, NAFBlock
from basicsr.models.archs.arch_util import MySequential
from basicsr.models.archs.local_arch import Local_Base
from einops import rearrange
from basicsr.models.archs.arch_util import MySequential, trunc_normal_


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c, dim,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None):
        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_ratio = overlap_ratio
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size


        self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.window_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + window_size  - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)

        # self.scale = c ** -0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(dim=-1)
    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ori + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ori - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
    def forward(self, x_l, x_r):
        #x_l B,C,H,W
        rpi = self.calculate_rpi_oca()

       
        b, c, h, w = x_l.shape
        #torch.Size([1, 32, 32, 32])
        # print('---------',x_l.shape)
        # print('---------',self.qkv(x_l).shape)
        #torch.Size([1, 32, 32, 96])
        # qkv = self.qkv(x_l).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
        # # print('---------qkv',qkv.shape) #torch.Size([3, 1, 64, 32, 32]
        # q = qkv[0].permute(0, 2, 3, 1) # b, h, w, c
        # # print('---------q',q.shape) #[1, 32, 32, 64]
        # kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, h, w
        # # print('---------kv',kv.shape) #[1, 128, 32, 32]
        # # partition windows
        # q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c #[16, 8,8, 64]
        # q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c
        # # print('---------q_windows',q_windows.shape)  #[16, 64, 64]

        # kv_windows = self.unfold(kv) # b, c*w*w, nw
        # # print('---------kv_windows',kv_windows.shape) #[1, 12288, 16]
        # kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        # # print('---------kv_windows',kv_windows.shape)
        # k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c


        q_windows = window_partition(self.norm_l(x_l), self.window_size)  # nw*b, window_size, window_size, c #[16, 8,8, 64]
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c
        # print('---------q_windows',q_windows.shape)  #[16, 64, 32]

        # kv_windows = self.unfold(kv) # b, c*w*w, nw
        # # print('---------kv_windows',kv_windows.shape) #[1, 8192, 20]
        # kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.window_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        # # print('---------kv_windows',kv_windows.shape) #[2, 20, 128, 32])
        # k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c
        # # print('---------k_windows',k_windows.shape) #([20, 128, 32])

        k_windows = self.unfold(x_r)
        print(k_windows.shape,x_r.shape)#torch.Size([1, 4096, 20]) torch.Size([1, 32, 32, 32])
        k_windows = rearrange(k_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=1, ch=c, owh=self.window_size, oww=self.overlap_win_size).contiguous() # 1, nw*b, ow*ow, c

        v_windows = self.unfold(x_r)
        v_windows = rearrange(v_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=1, ch=c, owh=self.window_size ,oww=self.overlap_win_size).contiguous() # 1, nw*b, ow*ow, c
        

        b_, nq, _ = q_windows.shape
        _, _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        print('============',k_windows.shape,b_, nq, self.num_heads, d,n)
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x_l_ = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        # x = x.view(b, h * w, self.dim)
########################

        b_r, c_r, h_r, w_r = x_r.shape
        # qkv_r = self.qkv(x_r).reshape(b_r, h_r, w_r, 3, c_r).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
        # q_r = qkv_r[0].permute(0, 2, 3, 1) # b, h, w, c
        # kv_r = torch.cat((qkv_r[1], qkv_r[2]), dim=1) # b, 2*c, h, w

        # # partition windows
        # q_windows_r = window_partition(q_r, self.window_size)  # nw*b, window_size, window_size, c
        # q_windows_r = q_windows_r.view(-1, self.window_size * self.window_size, c_r)  # nw*b, window_size*window_size, c

        # kv_windows_r = self.unfold(kv_r) # b, c*w*w, nw
        # kv_windows_r = rearrange(kv_windows_r, 'b_r (nc ch owh oww) nw -> nc (b_r nw) (owh oww) ch', nc=2, ch=c_r, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        # k_windows_r, v_windows_r = kv_windows_r[0], kv_windows_r[1] # nw*b, ow*ow, c


        q_windows_r = window_partition(self.norm_l(x_r), self.window_size)  # nw*b, window_size, window_size, c #[16, 8,8, 64]
        q_windows_r = q_windows_r.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c
      
        k_windows_r = self.unfold(x_l)
        # print(k_windows.shape)#torch.Size([1, 18432, 16])
        k_windows_r = rearrange(k_windows_r, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=1, ch=c_r, owh=self.window_size, oww=self.overlap_win_size).contiguous() # 1, nw*b, ow*ow, c

        v_windows_r = self.unfold(x_l)
        v_windows_r = rearrange(v_windows_r, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=1, ch=c_r, owh=self.window_size, oww=self.overlap_win_size).contiguous() # 1, nw*b, ow*ow, c
        
        b__r, nq_r, _ = q_windows_r.shape
        _, _, n_r, _ = k_windows_r.shape
        d_r = self.dim // self.num_heads
        q_r = q_windows_r.reshape(b__r, nq_r, self.num_heads, d_r).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k_r = k_windows_r.reshape(b__r, n_r, self.num_heads, d_r).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v_r = v_windows_r.reshape(b__r, n_r, self.num_heads, d_r).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q_r = q_r * self.scale
        attn_r = (q_r @ k_r.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn_r = attn_r + relative_position_bias.unsqueeze(0)

        attn_r = self.softmax(attn_r)
        attn_windows_r = (attn_r @ v_r).transpose(1, 2).reshape(b__r, nq_r, self.dim)

        # merge windows
        attn_windows_r = attn_windows_r.view(-1, self.window_size, self.window_size, self.dim)
        x_r_ = window_reverse(attn_windows_r, self.window_size, h_r, w_r)  # b h w c
        
        # print('===================',x_l_.shape)
        Q_l = self.l_proj1(self.norm_l(x_l_)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r_)).permute(0, 2, 1, 3) # B, H, c, W (transposed)



        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  #B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l) #B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_feats = tuple([x+factor*(new_x-x) for x, new_x in zip(feats, new_feats)])
        return new_feats

class NAFBlockSR(nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, dim,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = SCAM(c, dim,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias,
                qk_scale) if fusion else None

    def forward(self, *feats):
        feats = tuple([self.blk(x) for x in feats])
        if self.fusion:
            feats = self.fusion(*feats)
        return feats

class NAFNetSR(nn.Module):
    '''
    NAFNet for Super-Resolution
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0., fusion_from=-1, fusion_to=-1, dim=96,
                window_size=7,
                overlap_ratio=0.5,
                num_heads=(6,6,6,6),
                qkv_bias=True,
                qk_scale=None, dual=False):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate, 
                NAFBlockSR(
                    width, 
                    dim,
                    window_size,
                    overlap_ratio,
                    num_heads,
                    qkv_bias,
                    qk_scale,
                    fusion=(fusion_from <= i and i <= fusion_to), 
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]
        )

        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        if self.dual:
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats)
        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out

class NAFSSR_DSCAM_o(Local_Base, NAFNetSR):
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':
    num_blks = 128
    width = 128
    droppath=0.1
    train_size = (1, 6, 30, 90)

    net = NAFSSR_DSCAM(up_scale=2,train_size=train_size, fast_imp=True, width=width, num_blks=num_blks, drop_path_rate=droppath)

    inp_shape = (6, 64, 64)

    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # params = float(params[:-4])
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs, params)

    # from basicsr.models.archs.arch_util import measure_inference_speed
    # net = net.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(net, (data,))




