import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class BCELoss3D(nn.Module):
    def __init__(self):
        super(BCELoss3D, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, input, target):
        y_pred_flat = input.view(-1)  # 将预测值平展为一维向量
        y_true_flat = target.view(-1)  # 将真实标签平展为一维向量
        loss = self.bce_loss(y_pred_flat, y_true_flat)
        return loss.item()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss


class Dice_and_FocalLoss_BCE(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss_BCE, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)
        self.bce_loss = BCELoss3D()

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target) + self.bce_loss(input, target)
        return loss