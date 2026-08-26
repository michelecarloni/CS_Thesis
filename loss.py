import segmentation_models_pytorch as smp

dice_loss = smp.losses.DiceLoss(mode='multiclass')
focal_loss = smp.losses.FocalLoss(mode='multiclass')

def combined_loss(outputs, targets):
    return dice_loss(outputs, targets) + focal_loss(outputs, targets)