# Dice loss, BCE-Dice Loss
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ratio=(0.8,0.2)):
        super(DiceBCELoss, self).__init__()
        self.ratio_BCE, self.ratio_Dice = ratio

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)
        Dice_BCE_all = torch.zeros(inputs.shape[1])
        for channel in range(inputs.shape[1]):
        # flatten label and prediction tensors
            inputs_c = inputs[:,channel,:,:].reshape(-1)
            targets_c = targets[:,channel,:,:].reshape(-1)

            intersection = (inputs_c * targets_c).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (inputs_c.sum() + targets_c.sum() + smooth)
            BCE = F.binary_cross_entropy(inputs_c, targets_c, reduction='mean')
            Dice_BCE = self.ratio_BCE*BCE + self.ratio_Dice*dice_loss
            Dice_BCE_all[channel] = Dice_BCE

        return Dice_BCE_all.mean()

class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = F.one_hot(target, self.n_classes).permute(0,3,1,2).float()
        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()
