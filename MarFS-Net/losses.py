import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CrossEntropy', 'DiceLoss', 'DiceCELoss']

class CrossEntropy(nn.Module):
    # [修复] 增加 label_smoothing 参数
    def __init__(self, ignore_index=255, weight=None, label_smoothing=0.0):
        super().__init__()
        # 现在的 PyTorch CrossEntropyLoss 官方支持 label_smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=weight,
            label_smoothing=label_smoothing
        )

    def forward(self, input, target):
        return self.criterion(input, target)

class DiceLoss(nn.Module):
    def __init__(self, num_classes=8, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] -> Softmax
        inputs = F.softmax(inputs, dim=1)

        # targets: [B, H, W] -> One-Hot
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index)
            targets = targets * mask.long()

        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            mask = mask.unsqueeze(1).float()
            targets_one_hot = targets_one_hot * mask
            inputs = inputs * mask

        # Intersection & Union
        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        union = inputs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class DiceCELoss(nn.Module):
    # [修复] 这里的 label_smoothing 终于能传进去了
    def __init__(self, num_classes=8, ignore_index=255, weight=None, label_smoothing=0.0):
        super().__init__()
        self.ce = CrossEntropy(ignore_index=ignore_index, weight=weight, label_smoothing=label_smoothing)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, input, target):
        # 尺寸对齐 (防止 Deep Supervision 报错)
        if input.shape[-2:] != target.shape[-2:]:
            input = F.interpolate(input, size=target.shape[-2:], mode='bilinear', align_corners=True)

        loss_ce = self.ce(input, target.long())
        loss_dice = self.dice(input, target.long())
        # 0.5:0.5 是最稳健的黄金比例
        return 0.5 * loss_ce + 0.5 * loss_dice