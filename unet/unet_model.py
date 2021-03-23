from .unet_parts import *


class DualNorm_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, batchsize=4, nonlinear='relu', norm_type='BN',
                 spade_seg_mode='soft', spade_inferred_mode='mask', spade_aux_blocks='', spade_reduction=2):
        super(DualNorm_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.batchsize = batchsize

        block_names_list = ['inc', 'down1', 'down2', 'down3', 'down4', 'mid', 'up1', 'up2', 'up3', 'up4', 'outc']
        norm_types = {}
        for block_name in block_names_list:
            norm_types[block_name] = norm_type
        print(norm_types)

        spade_auxs = {}
        for block_name in block_names_list:
            if block_name in spade_aux_blocks:
                spade_auxs[block_name] = True
            else:
                spade_auxs[block_name] = False
        self.spade_auxs = spade_auxs
        # for k,v in spade_auxs:
        #     print(f'{k} spade aux: {v}')
        print(spade_auxs)
        self.inc = InConv(3, 32, n_classes=n_classes, block=ResBlock, batchsize=batchsize, nonlinear=nonlinear,
                          norm_type=norm_types['inc'], spade_seg_mode=spade_seg_mode, spade_inferred_mode=spade_inferred_mode,
                          spade_aux=spade_auxs['inc'], spade_reduction=spade_reduction, output_CHW=[32, 384, 384])
        self.down1 = Down(32, 64, n_classes=n_classes, block=ResBlock, batchsize=batchsize, nonlinear=nonlinear,
                          norm_type=norm_types['down1'], spade_seg_mode=spade_seg_mode, spade_inferred_mode=spade_inferred_mode,
                          spade_aux=spade_auxs['down1'], spade_reduction=spade_reduction, output_CHW=[64, 192, 192])
        self.down2 = Down(64, 128, n_classes=n_classes, block=ResBlock, batchsize=batchsize, nonlinear=nonlinear,
                          norm_type=norm_types['down2'], spade_seg_mode=spade_seg_mode, spade_inferred_mode=spade_inferred_mode,
                          spade_aux=spade_auxs['down2'], spade_reduction=spade_reduction, output_CHW=[128, 96, 96])
        self.down3 = Down(128, 256, n_classes=n_classes, block=ResBlock, batchsize=batchsize, nonlinear=nonlinear,
                          norm_type=norm_types['down3'], spade_seg_mode=spade_seg_mode, spade_inferred_mode=spade_inferred_mode,
                          spade_aux=spade_auxs['down3'], spade_reduction=spade_reduction, output_CHW=[256, 48, 48])
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512, n_classes=n_classes, block=ResBlock, batchsize=batchsize, nonlinear=nonlinear,
                          norm_type=norm_types['down4'], spade_seg_mode=spade_seg_mode, spade_inferred_mode=spade_inferred_mode,
                          spade_aux=spade_auxs['down4'], spade_reduction=spade_reduction, output_CHW=[512, 24, 24])
        self.mid = Mid(512, 512, n_classes=n_classes, batchsize=batchsize, nonlinear=nonlinear, norm_type=norm_types['mid'],
                       spade_seg_mode=spade_seg_mode, spade_inferred_mode=spade_inferred_mode, spade_aux=spade_auxs['mid'],
                       spade_reduction=spade_reduction, output_CHW=[512, 24, 24])
        self.up1 = Up(512, 256 // factor, n_classes=n_classes, bilinear=bilinear, block=ResBlock, batchsize=batchsize,
                      nonlinear=nonlinear, norm_type=norm_types['up1'], spade_seg_mode=spade_seg_mode,
                      spade_inferred_mode=spade_inferred_mode, spade_aux=spade_auxs['up1'], spade_reduction=spade_reduction,
                      output_CHW=[256, 48, 48])
        self.up2 = Up(256, 128 // factor, n_classes=n_classes, bilinear=bilinear, block=ResBlock, batchsize=batchsize,
                      nonlinear=nonlinear, norm_type=norm_types['up2'], spade_seg_mode=spade_seg_mode,
                      spade_inferred_mode=spade_inferred_mode, spade_aux=spade_auxs['up2'], spade_reduction=spade_reduction,
                      output_CHW=[128, 96, 96])
        self.up3 = Up(128, 64 // factor, n_classes=n_classes, bilinear=bilinear, block=ResBlock, batchsize=batchsize,
                      nonlinear=nonlinear, norm_type=norm_types['up3'], spade_seg_mode=spade_seg_mode,
                      spade_inferred_mode=spade_inferred_mode, spade_aux=spade_auxs['up3'], spade_reduction=spade_reduction,
                      output_CHW=[64, 192, 192])
        self.up4 = Up(64, 32, n_classes=n_classes, bilinear=bilinear, block=ResBlock, batchsize=batchsize,
                      nonlinear=nonlinear, norm_type=norm_types['up4'], spade_seg_mode=spade_seg_mode,
                      spade_inferred_mode=spade_inferred_mode, spade_aux=spade_auxs['up4'], spade_reduction=spade_reduction,
                      output_CHW=[32, 384, 384])
        self.outc = OutConv(32, n_classes, batchsize=batchsize, nonlinear=nonlinear, norm_type=norm_types['outc'], output_CHW=[32, 384, 384])


    def forward(self, x, seg=None):
        if seg is None:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5_1 = self.down4(x4)
            x = self.mid(x5_1)
            x = self.up1(x, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            x = self.outc(x)
            return x

        else:
            x1 = self.inc(x, seg=(seg if self.spade_auxs['inc'] else None))
            x2 = self.down1(x1, seg=(seg if self.spade_auxs['down1'] else None))
            x3 = self.down2(x2, seg=(seg if self.spade_auxs['down2'] else None))
            x4 = self.down3(x3, seg=(seg if self.spade_auxs['down3'] else None))
            x5_1 = self.down4(x4, seg=(seg if self.spade_auxs['down4'] else None))
            x = self.mid(x5_1, seg=(seg if self.spade_auxs['mid'] else None))
            x = self.up1(x, x4, seg=(seg if self.spade_auxs['up1'] else None))
            x = self.up2(x, x3, seg=(seg if self.spade_auxs['up2'] else None))
            x = self.up3(x, x2, seg=(seg if self.spade_auxs['up3'] else None))
            x = self.up4(x, x1, seg=(seg if self.spade_auxs['up4'] else None))
            x = self.outc(x, seg=None)
            return x
#
#

