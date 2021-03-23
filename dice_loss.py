import torch
from torch.autograd import Function

import torch.nn.functional as F
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, target, num_classes=2, return_hard_dice=False, softmax=False):
        """
        input  : NxCxHxW Variable
        target :  NxHxW LongTensor

        """
        assert inputs.dim() == 4, "Input must be a 4D Tensor."
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).contiguous()
        assert inputs.size() == target_onehot.size(), "Input sizes must be equal."

        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        # binary dice
        if inputs.shape[1] == 2:
            intersection = torch.sum(inputs * target_onehot, [2, 3])
            L = torch.sum(inputs, [2, 3])
            R = torch.sum(target_onehot, [2, 3])
            coefficient = (2 * intersection + 1e-6) / (L + R + 1e-6)
            loss = 1 - coefficient.mean()

        else:
            intersection = torch.sum(inputs * target_onehot, (0, 2, 3))
            L = torch.sum(inputs * inputs, (0, 2, 3))
            R = torch.sum(target_onehot * target_onehot, (0, 2, 3))

            coefficient = (2 * intersection + 1e-6) / (L + R + 1e-6)
            loss = 1 - coefficient.mean()

        if return_hard_dice:
            pred_hard = torch.argmax(inputs, dim=1)
            pred_onehot = F.one_hot(pred_hard, num_classes=num_classes).permute(0, 3, 1, 2).contiguous()
            hard_intersection = torch.sum(pred_onehot * target_onehot, (0, 2, 3))
            hard_cardinality = torch.sum(pred_onehot + target_onehot, (0, 2, 3))
            hard_dice_loss = 1 - (2 * hard_intersection + 1e-6) / (hard_cardinality + 1e-6)
            hard_dice_loss = hard_dice_loss.mean()
            return loss, hard_dice_loss
        else:
            return loss



