import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class OriginalCombinedLoss(nn.Module):
    def __init__(self, num_classes=None, bg_weight=0.15):
        """
        Weighted Cross-Entropy + Foreground Multi-Class Dice Loss.
        - DiceLoss: Optimizes only the target subset crops ( ignores Class 0 )
          so the massive background class doesn't stall the Dice score.
        - CrossEntropyLoss: Evaluates all classes (0..K) using a lower weight
          for Class 0 (bg_weight=0.15) and 1.0 for target crops (1..K).
        """
        super().__init__()
        self.bg_weight = bg_weight
        
        # Dice Loss ignores Class 0 (Background) and optimizes only your subset crops
        self.dice_loss = smp.losses.DiceLoss(
            mode='multiclass',
            from_logits=True,
            ignore_index=0,
            eps=1e-5
        )
        
        # If num_classes is provided at init, pre-build the weight tensor
        if num_classes is not None:
            weights = torch.ones(num_classes, dtype=torch.float32)
            weights[0] = bg_weight
            self.register_buffer("class_weights", weights)
        else:
            self.class_weights = None

    def forward(self, outputs, targets):
        # Cast to float32 for numerical stability
        outputs = outputs.float()
        
        # Dynamically create the weight tensor [bg_weight, 1.0, 1.0, ...] if not passed at init
        if self.class_weights is None or self.class_weights.device != outputs.device:
            num_classes = outputs.shape[1]
            weights = torch.ones(num_classes, dtype=torch.float32, device=outputs.device)
            weights[0] = self.bg_weight
            self.class_weights = weights

        # 1. Foreground Crop Dice Loss (ignores Class 0)
        dice = self.dice_loss(outputs, targets)
        
        # 2. Weighted Cross-Entropy Loss (penalizes false positives on Class 0 with bg_weight)
        ce = nn.functional.cross_entropy(
            outputs, 
            targets, 
            weight=self.class_weights
        )
        
        return dice + ce