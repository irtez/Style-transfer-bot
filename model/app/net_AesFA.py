import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple

class AesFA_test(nn.Module):
    def __init__(self):
        super(AesFA_test, self).__init__()
        alpha_in = 0.5
        alpha_out = 0.5
        sk = 3
        
        self.netE = Encoder(in_dim=3, nf=64, style_kernel=(sk, sk), alpha_in=alpha_in, alpha_out=alpha_out)
        self.netS = Encoder(in_dim=3, nf=64, style_kernel=(sk, sk), alpha_in=alpha_in, alpha_out=alpha_out)
        self.netG = Decoder(nf=64, out_dim=3, style_channel=256, style_kernel=(sk, sk, 3), alpha_in=alpha_in, freq_ratio=(1, 1), alpha_out=alpha_out)


    def forward(self, real_A: Tensor, real_B: Tensor) -> Tensor:
        with torch.no_grad():
            content_A = self.netE.forward_test(real_A, 'content')
            style_B = self.netS.forward_test(real_B, 'style')
            trs_AtoB = self.netG.forward_test(content_A, style_B)
        return trs_AtoB


class Encoder(nn.Module):    
    def __init__(self, in_dim, nf=64, style_kernel=[3, 3], alpha_in=0.5, alpha_out=0.5):
        super(Encoder, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_dim, out_channels=nf, kernel_size=7, stride=1, padding=3)        
        
        self.OctConv1_1 = OctConvFirst(in_channels=nf, out_channels=nf, kernel_size=3, stride=2, padding=1, groups=64, alpha_in=alpha_in, alpha_out=alpha_out, type="first")       
        self.OctConv1_2 = OctConvNormal(in_channels=nf, out_channels=2*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv1_3 = OctConvNormal(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        
        self.OctConv2_1 = OctConvNormal(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=2, padding=1, groups=128, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConvNormal(in_channels=2*nf, out_channels=4*nf, kernel_size=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_3 = OctConvNormal(in_channels=4*nf, out_channels=4*nf, kernel_size=3, stride=1, padding=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")

        self.pool_h = nn.AdaptiveAvgPool2d((style_kernel[0], style_kernel[0]))
        self.pool_l = nn.AdaptiveAvgPool2d((style_kernel[1], style_kernel[1]))
        
        self.relu = nn.LeakyReLU()
    
    def forward_test(self, x: Tensor, cond: str) -> Tuple[Tensor, Tensor]:
        out = self.conv(x)   
        
        out = self.OctConv1_1(out)
        out = self.relu(out[0]), self.relu(out[1])
        out = self.OctConv1_2(out)
        out = self.relu(out[0]), self.relu(out[1])
        out = self.OctConv1_3(out)
        out = self.relu(out[0]), self.relu(out[1])
        
        out = self.OctConv2_1(out)   
        out = self.relu(out[0]), self.relu(out[1])
        out = self.OctConv2_2(out)
        out = self.relu(out[0]), self.relu(out[1])
        out = self.OctConv2_3(out)
        out = self.relu(out[0]), self.relu(out[1])
        
        if cond == 'style':
            out_high, out_low = out[0], out[1]
            out_sty_h = self.pool_h(out_high)
            out_sty_l = self.pool_l(out_low)
            return out_sty_h, out_sty_l
        else:
            return out

class Decoder(nn.Module):
    def __init__(self, nf=64, out_dim=3, style_channel=512, style_kernel=[3, 3, 3], alpha_in=0.5, alpha_out=0.5, freq_ratio=[1,1], pad_type='reflect'):
        super(Decoder, self).__init__()

        group_div = [1, 2, 4, 8]
        self.up_oct = nn.Upsample(scale_factor=2)

        self.AdaOctConv1_1 = AdaOctConv(in_channels=4*nf, out_channels=4*nf, group_div=group_div[0], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=4*nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv1_2 = OctConvNormal(in_channels=4*nf, out_channels=2*nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_1 = Oct_Conv_aftup(in_channels=2*nf, out_channels=2*nf, kernel_size=3, stride=1, padding=1, pad_type=pad_type, alpha_in=alpha_in, alpha_out=alpha_out)

        self.AdaOctConv2_1 = AdaOctConv(in_channels=2*nf, out_channels=2*nf, group_div=group_div[1], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=2*nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv2_2 = OctConvNormal(in_channels=2*nf, out_channels=nf, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.oct_conv_aftup_2 = Oct_Conv_aftup(nf, nf, 3, 1, 1, pad_type, alpha_in, alpha_out)

        self.AdaOctConv3_1 = AdaOctConv(in_channels=nf, out_channels=nf, group_div=group_div[2], style_channels=style_channel, kernel_size=style_kernel, stride=1, padding=1, oct_groups=nf, alpha_in=alpha_in, alpha_out=alpha_out, type="normal")
        self.OctConv3_2 = OctConvLast(in_channels=nf, out_channels=nf//2, kernel_size=1, stride=1, alpha_in=alpha_in, alpha_out=alpha_out, type="last", freq_ratio=freq_ratio)
       
        self.conv4 = nn.Conv2d(in_channels=nf//2, out_channels=out_dim, kernel_size=1)
    
    def forward_test(self, content: Tensor, style: Tensor) -> Tensor:
        out = self.AdaOctConv1_1(content, style)
        out = self.OctConv1_2(out)
        out = self.up_oct(out[0]), self.up_oct(out[1])
        out = self.oct_conv_aftup_1(out)

        out = self.AdaOctConv2_1(out, style)
        out = self.OctConv2_2(out)
        out = self.up_oct(out[0]), self.up_oct(out[1])
        out = self.oct_conv_aftup_2(out)
     
        out = self.AdaOctConv3_1(out, style)
        out = self.OctConv3_2(out)

        out = self.conv4(out[0])
        return out



class OctConvFirst(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, pad_type='reflect', alpha_in=0.5, alpha_out=0.5, type='normal', freq_ratio = (1, 1)):
        super(OctConvFirst, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_ch_in = int(in_channels * (1 - self.alpha_in))
        hf_ch_out = int(out_channels * (1 -self. alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.is_dw = groups == in_channels
        self.convh = nn.Conv2d(in_channels, hf_ch_out, kernel_size=kernel_size,
                                stride=stride, padding=padding, padding_mode=pad_type, bias = False)
        self.convl = nn.Conv2d(in_channels, lf_ch_out,
                                kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        hf = self.convh(x)
        lf = self.avg_pool(x)
        lf = self.convl(lf)
        return hf, lf
    
class OctConvLast(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, pad_type='reflect', alpha_in=0.5, alpha_out=0.5, type='normal', freq_ratio = (1, 1)):
        
        super(OctConvLast, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_ch_in = int(in_channels * (1 - self.alpha_in))
        hf_ch_out = int(out_channels * (1 -self. alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.is_dw = groups == in_channels
        self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
    
    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        hf, lf = x[0], x[1]
        out_h = self.convh(hf)
        out_l = self.convl(self.upsample(lf))
        output = out_h * self.freq_ratio[0] + out_l * self.freq_ratio[1]
        return output, out_h, out_l
    
class OctConvNormal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, groups=1, pad_type='reflect', alpha_in=0.5, alpha_out=0.5, type='normal', freq_ratio = (1, 1)):
        super(OctConvNormal, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_ch_in = int(in_channels * (1 - self.alpha_in))
        hf_ch_out = int(out_channels * (1 -self. alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.is_dw = groups == in_channels

        self.L2L = nn.Conv2d(
            lf_ch_in, lf_ch_out,
            kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(alpha_in * groups), padding_mode=pad_type, bias=False
        )
        self.H2H = nn.Conv2d(
            hf_ch_in, hf_ch_out,
            kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(groups - alpha_in * groups), padding_mode=pad_type, bias=False
        )
        if self.is_dw:
            self.L2H = None
            self.H2L = None
        else:
            self.L2H = nn.Conv2d(
                lf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
            )
            self.H2L = nn.Conv2d(
                hf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
            )
    
    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        hf, lf = x[0], x[1]
        if self.L2H is None and self.H2L is None:
            hf, lf = self.H2H(hf), self.L2L(lf)
        else:
            hf_l = Tensor(data=hf.shape) / 2
            lf_l = Tensor(data=lf.shape)
            while torch.ne(hf_l, lf_l)[-1]:
                if torch.gt(hf_l, lf_l)[-1]:
                    hf = hf[:, :, :, :-1]
                else:
                    lf = lf[:, :, :, :-1]
                hf_l = Tensor(data=hf.shape) / 2
                lf_l = Tensor(data=lf.shape)
            while torch.ne(hf_l, lf_l)[-2]:
                if torch.gt(hf_l, lf_l)[-2]:
                    hf = hf[:, :, :-1, :]
                else:
                    lf = lf[:, :, :-1, :]
                hf_l = Tensor(data=hf.shape) / 2
                lf_l = Tensor(data=lf.shape)
            hf, lf = self.H2H(hf) + self.L2H(self.upsample(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))
        return hf, lf, lf

    
class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size):
        super(KernelPredictor, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.w_channels = style_channels
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * out_channels // n_groups,
                                 kernel_size=kernel_size,
                                 padding=(math.ceil(padding), math.ceil(padding)),
                                 padding_mode='reflect')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels * out_channels // n_groups,
                      kernel_size=1)
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels,
                      kernel_size=1)
        )

    def forward(self, w: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(w.shape[0],
                                      self.out_channels,
                                      self.in_channels // self.n_groups,
                                      self.kernel_size, self.kernel_size)

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(w.shape[0],
                                          self.out_channels,
                                          self.out_channels // self.n_groups,
                                          1, 1)
        bias = self.bias(w)
        bias = bias.reshape(w.shape[0], self.out_channels)
        return w_spatial, w_pointwise, bias

class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super(AdaConv2d, self).__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(math.ceil(padding), math.floor(padding)),
                              padding_mode='reflect')

    def forward(self, x: Tensor, w_spatial: Tensor, w_pointwise: Tensor, bias: Tensor) -> Tensor:
        x = F.instance_norm(x)

        ys = [0] * x.shape[0]
        for i in range(x.shape[0]):
            y = self.forward_single(x[i:i+1], w_spatial[i], w_pointwise[i], bias[i])
            ys[i] = y.clone()
        ys = torch.cat(ys, dim=1)
        ys = self.conv(ys)
        return ys

    def forward_single(self, x: Tensor, w_spatial: Tensor, w_pointwise: Tensor, bias: Tensor) -> Tensor:
        padding = (w_spatial.shape[-1] - 1) / 2
        pad = (math.ceil(padding), math.floor(padding), math.ceil(padding), math.floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x
    
class AdaOctConv(nn.Module):
    def __init__(self, in_channels, out_channels, group_div, style_channels, kernel_size,
                 stride, padding, oct_groups, alpha_in, alpha_out, type='normal'):
        super(AdaOctConv, self).__init__()
        self.in_channels = in_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.type = type
        
        h_in = int(in_channels * (1 - self.alpha_in))
        l_in = in_channels - h_in

        n_groups_h = h_in // group_div
        n_groups_l = l_in // group_div
        
        style_channels_h = int(style_channels * (1 - self.alpha_in))
        style_channels_l = int(style_channels - style_channels_h)
        
        kernel_size_h = kernel_size[0]
        kernel_size_l = kernel_size[1]
        kernel_size_A = kernel_size[2]

        self.kernelPredictor_h = KernelPredictor(in_channels=h_in,
                                              out_channels=h_in,
                                              n_groups=n_groups_h,
                                              style_channels=style_channels_h,
                                              kernel_size=kernel_size_h)
        self.kernelPredictor_l = KernelPredictor(in_channels=l_in,
                                               out_channels=l_in,
                                               n_groups=n_groups_l,
                                               style_channels=style_channels_l,
                                               kernel_size=kernel_size_l)
        
        self.AdaConv_h = AdaConv2d(in_channels=h_in, out_channels=h_in, n_groups=n_groups_h)
        self.AdaConv_l = AdaConv2d(in_channels=l_in, out_channels=l_in, n_groups=n_groups_l)
        
        self.OctConv = OctConvNormal(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size_A, stride=stride, padding=padding, groups=oct_groups,
                            alpha_in=alpha_in, alpha_out=alpha_out, type=type)
        
        self.relu = nn.LeakyReLU()

    def forward(self, content: Tuple[Tensor, Tensor], style: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        c_hf, c_lf = content[0], content[1]
        s_hf, s_lf = style[0], style[1]
        k_h = self.kernelPredictor_h(s_hf)
        h_w_spatial, h_w_pointwise, h_bias = k_h[0], k_h[1], k_h[2]
        k_l = self.kernelPredictor_l(s_lf)
        l_w_spatial, l_w_pointwise, l_bias = k_l[0], k_l[1], k_l[2]
        
        output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
        output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
        output = self.relu(output_h), self.relu(output_l)
        output = self.OctConv(output)
        output = self.relu(output[0]), self.relu(output[1]), self.relu(output[2])
        return output
        
class Oct_Conv_aftup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pad_type, alpha_in, alpha_out):
        super(Oct_Conv_aftup, self).__init__()
        lf_in = int(in_channels*alpha_in)
        lf_out = int(out_channels*alpha_out)
        hf_in = in_channels - lf_in
        hf_out = out_channels - lf_out

        self.conv_h = nn.Conv2d(in_channels=hf_in, out_channels=hf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        self.conv_l = nn.Conv2d(in_channels=lf_in, out_channels=lf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
    
    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        hf, lf = x[0], x[1]
        hf = self.conv_h(hf)
        lf = self.conv_l(lf)
        return hf, lf
