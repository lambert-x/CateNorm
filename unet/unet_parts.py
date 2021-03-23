""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spade import SPADE


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes=2, mid_channels=None, batchsize=4,
                 nonlinear='relu',
                 norm_type='BN', spade_seg_mode='soft', spade_inferred_mode='mask', spade_aux=False,
                 spade_reduction=2, output_CHW=None):
        super().__init__()
        nonlinear_dict = nn.ModuleDict({
            'relu': nn.ReLU(inplace=True),
        })
        self.nonlinear_layer = nonlinear_dict[nonlinear]
        norm_dict = {
            'BN': nn.BatchNorm2d,
            'IN': nn.InstanceNorm2d,
            'GN': nn.GroupNorm,
            'LN': nn.LayerNorm,
        }
        self.norm_type = norm_type
        self.norm_layer = norm_dict[norm_type]

        if not mid_channels:
            mid_channels = out_channels

        if self.norm_type == 'LN':
            self.norm_0 = self.norm_layer(output_CHW)
            self.norm_1 = self.norm_layer(output_CHW)
        elif self.norm_type == 'GN':
            self.norm_0 = self.norm_layer(16, in_channels)
            self.norm_1 = self.norm_layer(16, mid_channels)
        else:
            self.norm_0 = self.norm_layer(in_channels)
            self.norm_1 = self.norm_layer(mid_channels)

        self.spade_aux = spade_aux

        if spade_aux:
            self.norm_0_aux = SPADE(in_channels, n_classes, reduction=spade_reduction, spade_seg_mode=spade_seg_mode,
                                    spade_inferred_mode=spade_inferred_mode)
            self.norm_1_aux = SPADE(mid_channels, n_classes, reduction=spade_reduction, spade_seg_mode=spade_seg_mode,
                                    spade_inferred_mode=spade_inferred_mode)

        self.conv_0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, seg=None):
        identity = x

        if seg is None:
            x = self.conv_0(self.nonlinear_layer(self.norm_0(x)))
            out = self.conv_1(self.nonlinear_layer(self.norm_1(x)))
        else:
            x = self.conv_0(self.nonlinear_layer(self.norm_0_aux(x, seg)))
            out = self.conv_1(self.nonlinear_layer(self.norm_1_aux(x, seg)))

        out += identity
        return out


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes=2, block=ResBlock, batchsize=4,
                 nonlinear='relu',
                 norm_type='BN',
                 spade_seg_mode='soft', spade_inferred_mode='mask',
                 spade_aux=False, spade_reduction=2, output_CHW=None):
        super(InConv, self).__init__()
        nonlinear_dict = nn.ModuleDict({
            'relu': nn.ReLU(inplace=True),
        })
        self.nonlinear_layer = nonlinear_dict[nonlinear]
        norm_dict = {
            'BN': nn.BatchNorm2d,
            'IN': nn.InstanceNorm2d,
            'GN': nn.GroupNorm,
            'LN': nn.LayerNorm
        }
        self.norm_layer = norm_dict[norm_type]
        self.norm_type = norm_type
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.spade_aux = spade_aux
        self.block = block(out_channels, out_channels, n_classes=n_classes, batchsize=batchsize,
                           nonlinear=nonlinear, norm_type=norm_type, spade_seg_mode=spade_seg_mode,
                           spade_inferred_mode=spade_inferred_mode, spade_aux=spade_aux,
                           spade_reduction=spade_reduction, output_CHW=output_CHW)

    def forward(self, x, seg=None):
        x = self.conv(x)
        if self.spade_aux:
            x = self.block(x, seg=seg)
        else:
            x = self.block(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, n_classes=2, block=ResBlock, batchsize=4,
                 nonlinear='relu', norm_type='BN', spade_seg_mode='soft', spade_inferred_mode='mask', spade_aux=False,
                 spade_reduction=2, output_CHW=None):
        super().__init__()
        nonlinear_dict = nn.ModuleDict({
            'relu': nn.ReLU(inplace=True),
        })
        self.nonlinear_layer = nonlinear_dict[nonlinear]

        norm_dict = {
            'BN': nn.BatchNorm2d,
            'IN': nn.InstanceNorm2d,
            'GN': nn.GroupNorm,
            'LN': nn.LayerNorm
        }
        self.norm_layer = norm_dict[norm_type]
        self.norm_type = norm_type
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.blockname = block
        self.spade_aux = spade_aux
        self.block = block(out_channels, out_channels, n_classes=n_classes, batchsize=batchsize,
                           nonlinear=nonlinear, norm_type=norm_type, spade_seg_mode=spade_seg_mode,
                           spade_inferred_mode=spade_inferred_mode, spade_aux=spade_aux,
                           spade_reduction=spade_reduction, output_CHW=output_CHW)

    def forward(self, x, seg=None):
        x = self.maxpool_conv(x)
        if self.spade_aux:
            x = self.block(x, seg=seg)
        else:
            x = self.block(x)
        return x


class Mid(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, n_classes=2, mid_channels=None, batchsize=4,
                 nonlinear='relu', norm_type='BN', spade_seg_mode='soft', spade_inferred_mode='mask', spade_aux=False,
                 spade_reduction=2, output_CHW=None):
        super().__init__()
        self.block = ResBlock(in_channels, out_channels, n_classes, mid_channels, batchsize=batchsize,
                              nonlinear=nonlinear, norm_type=norm_type, spade_seg_mode=spade_seg_mode,
                              spade_inferred_mode=spade_inferred_mode, spade_aux=spade_aux,
                              spade_reduction=spade_reduction, output_CHW=output_CHW)
        self.spade_aux = spade_aux

    def forward(self, x, seg=None):
        if self.spade_aux:
            out = self.block(x, seg=seg)
        else:
            out = self.block(x)
        return out


class TransConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, batchsize=4, nonlinear='relu', norm_type='BN', output_CHW=None):
        super().__init__()
        nonlinear_dict = nn.ModuleDict({
            'relu': nn.ReLU(inplace=True),
        })
        self.nonlinear_layer = nonlinear_dict[nonlinear]
        norm_dict = {
            'BN': nn.BatchNorm2d,
            'IN': nn.InstanceNorm2d,
            'GN': nn.GroupNorm,
            'LN': nn.LayerNorm
        }
        self.norm_layer = norm_dict[norm_type]
        self.norm_type = norm_type
        if norm_type == 'GN':
            self.up = nn.Sequential(
                self.norm_layer(16, in_channels),
                self.nonlinear_layer,
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
        elif norm_type == 'LN':
            self.up = nn.Sequential(
                self.norm_layer([output_CHW[0] * 2, output_CHW[1] // 2, output_CHW[2] // 2]),
                self.nonlinear_layer,
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
        else:
            self.up = nn.Sequential(
                self.norm_layer(in_channels),
                self.nonlinear_layer,
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
        self.batchsize = batchsize

    def forward(self, x, seg=None):
        out = self.up(x)
        return out


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, n_classes=2, bilinear=True, block=ResBlock, batchsize=4,
                 nonlinear='relu', norm_type='BN', spade_seg_mode='soft', spade_inferred_mode='mask',
                 spade_aux=False, spade_reduction=2, output_CHW=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        nonlinear_dict = nn.ModuleDict({
            'relu': nn.ReLU(inplace=True),
        })

        self.nonlinear_layer = nonlinear_dict[nonlinear]
        norm_dict = {
            'BN': nn.BatchNorm2d,
            'IN': nn.InstanceNorm2d,
            'GN': nn.GroupNorm,
            'LN': nn.LayerNorm
        }
        self.norm_layer = norm_dict[norm_type]
        self.norm_type = norm_type
        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = block(in_channels, out_channels, in_channels // 2, nonlinear=nonlinear, norm_type=norm_type,
                              spade_seg_mode=spade_seg_mode, spade_inferred_mode=spade_inferred_mode,
                              spade_aux=spade_aux, spade_reduction=spade_reduction, output_CHW=output_CHW)
        else:
            self.up = TransConv(in_channels, batchsize=batchsize, nonlinear=nonlinear,
                                norm_type=norm_type, output_CHW=output_CHW)

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.blockname = block
            self.block = block(out_channels, out_channels, n_classes=n_classes, batchsize=batchsize,
                               nonlinear=nonlinear, norm_type=norm_type, spade_seg_mode=spade_seg_mode,
                               spade_inferred_mode=spade_inferred_mode, spade_aux=spade_aux,
                               spade_reduction=spade_reduction, output_CHW=output_CHW)
            self.spade_aux = spade_aux

    def forward(self, x1, x2, seg=None):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.spade_aux:
            x = self.block(x, seg=seg)
        else:
            x = self.block(x)

        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, batchsize=4, nonlinear='relu',
                 norm_type='BN', output_CHW=None):
        super(OutConv, self).__init__()
        nonlinear_dict = nn.ModuleDict({
            'relu': nn.ReLU(inplace=True),
        })
        self.nonlinear_layer = nonlinear_dict[nonlinear]
        norm_dict = {
            'BN': nn.BatchNorm2d,
            'IN': nn.InstanceNorm2d,
            'GN': nn.GroupNorm,
            'LN': nn.LayerNorm
        }
        self.norm_layer = norm_dict[norm_type]
        self.norm_type = norm_type
        if norm_type == 'GN':
            self.conv = nn.Sequential(
                self.norm_layer(16, in_channels),
                self.nonlinear_layer,
                nn.Conv2d(in_channels, out_channels, kernel_size=1))
        elif norm_type == 'LN':
            self.conv = nn.Sequential(
                self.norm_layer(output_CHW),
                self.nonlinear_layer,
                nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            self.conv = nn.Sequential(
                self.norm_layer(in_channels),
                self.nonlinear_layer,
                nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x, seg=None):
        return self.conv(x)
