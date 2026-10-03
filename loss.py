import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, ignore_index=0, gamma=2.0):
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