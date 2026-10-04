import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class OriginalCombinedLoss(nn.Module):
    def __init__(self):
        """
        The original combined loss (Dice + Focal) that evaluates all pixels, 
        including the background (Class 0). No frequency weighting or masking.
        """
        super().__init__()
        # SMP Dice and Focal losses without an ignore_index.
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', eps=1e-5)
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass')

    def forward(self, outputs, targets):
        # Cast to float32 to prevent FP16 overflow during AMP
        outputs = outputs.float()
        return self.dice_loss(outputs, targets) + self.focal_loss(outputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, ignore_index=0, gamma=2.0):
        """
        Custom Combined Loss for imbalanced segmentation.
        - DiceLoss (SMP): Excludes the background class automatically.
        - Custom Weighted Focal Loss: Natively supports 1D tensor alpha weights 
          and correctly isolates target probabilities.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.alpha = alpha
        
        # INCREASED EPSILON: 1e-5 prevents FP16 rounding to 0.0 during AMP training
        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass', 
            ignore_index=ignore_index, 
            eps=1e-5
        )

    def forward(self, outputs, targets):
        # CAST TO FP32: Safely calculate loss in 32-bit precision to prevent overflow
        outputs = outputs.float()
        
        # 1. SMP Dice Loss
        dice = self.dice_loss(outputs, targets)
        
        # 2. Native PyTorch Weighted Focal Loss
        probs = F.softmax(outputs, dim=1)
        
        targets_masked = targets.clone()
        targets_masked[targets == self.ignore_index] = 0  
        pt = probs.gather(1, targets_masked.unsqueeze(1)).squeeze(1)
        
        ce_loss = F.cross_entropy(
            outputs, 
            targets, 
            weight=self.alpha, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        
        focal_loss_unreduced = ((1 - pt) ** self.gamma) * ce_loss
        
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() > 0:
            focal = focal_loss_unreduced[valid_mask].mean()
        else:
            focal = torch.tensor(0.0, device=outputs.device)
            
        return dice + focal