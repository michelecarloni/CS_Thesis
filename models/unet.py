import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, encoder_name="resnet18", encoder_weights=None, encoder_depth=3):
        """
        A custom U-Net wrapper optimized for small 32x32 tiles.
        
        Parameters:
        - in_channels: Number of input bands (e.g., 218 for Hyperspectral, 10 for Multispectral).
        - num_classes: Number of distinct crops in your current subset + 1 (for Background).
        - encoder_name: The backbone architecture (default is a lightweight resnet18).
        - encoder_weights: Keep as None for non-RGB spectral data.
        - encoder_depth: Limited to 3 to prevent over-compressing the 32x32 tile.
        """
        super(UNet, self).__init__()
        
        # Slice the default decoder_channels to match the custom encoder_depth
        default_decoder_channels = (256, 128, 64, 32, 16)
        adjusted_decoder_channels = default_decoder_channels[:encoder_depth]

        # Initialize the underlying SMP model
        self.model = smp.Unet(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,  
            in_channels=in_channels,          
            classes=num_classes,              
            encoder_depth=encoder_depth,      
            decoder_channels=adjusted_decoder_channels 
        )

    def forward(self, x):
        """
        Forward pass. 
        Input shape: (Batch, Channels, 32, 32)
        Output shape: (Batch, num_classes, 32, 32)
        """
        return self.model(x)