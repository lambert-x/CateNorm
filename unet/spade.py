import torch.nn as nn
import torch.nn.functional as F
import torch

class SPADE(nn.Module):
    def __init__(self, norm_nc, n_classes, reduction=2, ks=3, param_free_norm_type='BN', spade_seg_mode='soft',
                 spade_inferred_mode='mask'):
        super().__init__()
        pw = ks // 2
        nhidden = norm_nc // reduction

        if param_free_norm_type == 'BN':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)


        # define mlp_shared
        if spade_inferred_mode == 'mask':
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(n_classes, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU()
            )



        # define mlp_gamma/beta

        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.n_classes = n_classes
        self.spade_seg_mode = spade_seg_mode


        self.spade_inferred_mode = spade_inferred_mode


    def forward(self, x, segmap=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        if self.spade_inferred_mode == 'mask':
            if self.spade_seg_mode == 'hard':
                # segmap = (segmap > 0.5).float()
                segmap = F.interpolate(segmap.float(), size=x.size()[2:], mode='nearest')
            elif self.spade_seg_mode == 'soft':
                segmap = F.interpolate(segmap.float(), size=x.size()[2:], mode='bilinear', align_corners=True)
            actv = self.mlp_shared(segmap)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out
