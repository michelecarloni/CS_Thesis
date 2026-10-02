import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

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
        
        # SMP DiceLoss handles ignore_index natively
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)

    def forward(self, outputs, targets):
        # 1. SMP Dice Loss
        dice = self.dice_loss(outputs, targets)
        
        # 2. Native PyTorch Weighted Focal Loss
        # Get raw probabilities across all classes
        probs = F.softmax(outputs, dim=1)
        
        # Safely extract the probabilities specifically for the true target classes
        targets_masked = targets.clone()
        targets_masked[targets == self.ignore_index] = 0  # Safe fallback for ignored indices
        pt = probs.gather(1, targets_masked.unsqueeze(1)).squeeze(1)
        
        # Calculate standard Cross Entropy applying our frequency weights and ignoring background
        ce_loss = F.cross_entropy(
            outputs, 
            targets, 
            weight=self.alpha, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        
        # Apply the Focal factor: (1 - pt)^gamma
        focal_loss_unreduced = ((1 - pt) ** self.gamma) * ce_loss
        
        # Average the loss ONLY over valid (non-background) pixels
        valid_mask = (targets != self.ignore_index)
        if valid_mask.sum() > 0:
            focal = focal_loss_unreduced[valid_mask].mean()
        else:
            focal = torch.tensor(0.0, device=outputs.device)
            
        return dice + focal