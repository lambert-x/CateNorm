import torch
from torch.autograd import Function

import torch.nn.functional as F
import torch.nn as nn


class Weighted_Cross_Entropy_Loss(nn.Module):
    def __init__(self):
        super(Weighted_Cross_Entropy_Loss, self).__init__()

    def forward(self, inputs, target, num_classes=2, weighted=False, softmax=False):
        """
        input  : NxCxHxW Variable
        target :  NxHxW LongTensor
        """
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).contiguous()
        if weighted:
            ce_weight = 1 - (torch.sum(target_onehot, dim=(0, 2, 3)).float() / torch.sum(target_onehot).float())
            ce_weight = ce_weight.view(1, num_classes, 1, 1)
        else:
            ce_weight = 1
        loss = - 1.0 * torch.sum(ce_weight * target_onehot * torch.log(inputs.clamp(min=0.005, max=1)), dim=1)
        loss = loss.mean()
        return loss
