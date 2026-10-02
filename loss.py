import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, ignore_index=0):
        """
        Custom Combined Loss for imbalanced segmentation.
        - DiceLoss: Excludes the background class from the intersection/union math entirely.
        - FocalLoss: Excludes the background and weights the crop classes using the alpha tensor.
        """
        super().__init__()
        
        # Initialize SMP losses with the background ignored
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', alpha=alpha, ignore_index=ignore_index)

    def forward(self, outputs, targets):
        return self.dice_loss(outputs, targets) + self.focal_loss(outputs, targets)