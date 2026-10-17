import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SpectralSEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention block to weight informative 
    spectral bands and suppress noisy bands before spatial downsampling.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid_channels = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class UNet(nn.Module):
    def __init__(
        self, 
        in_channels, 
        num_classes, 
        encoder_name="resnet18", 
        encoder_weights=None, 
        encoder_depth=4
    ):
        """
        Custom U-Net wrapper with ResNet-18 encoder, global per-band BatchNorm,
        and a Spectral Attention Stem for Multispectral (10ch) & Hyperspectral (218ch) tiles.
        """
        super(UNet, self).__init__()
        
        # 1. Global Per-Channel Normalization (replaces destructive per-tile normalization)
        self.input_norm = nn.BatchNorm2d(in_channels)
        
        # 2. Spectral Adapter Stem: prevents crushing 218 hyperspectral bands directly in a 7x7 stride-2 conv
        stem_channels = 64 if in_channels > 16 else in_channels
        if in_channels > 16:
            self.spectral_stem = nn.Sequential(
                SpectralSEBlock(in_channels, reduction=8),
                nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.SiLU(inplace=True)
            )
        else:
            self.spectral_stem = SpectralSEBlock(in_channels, reduction=2)

        # 3. Slice default decoder_channels to match encoder_depth
        default_decoder_channels = (256, 128, 64, 32, 16)
        adjusted_decoder_channels = default_decoder_channels[:encoder_depth]

        # 4. Initialize underlying SMP U-Net
        self.model = smp.Unet(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,  
            in_channels=stem_channels,          
            classes=num_classes,              
            encoder_depth=encoder_depth,      
            decoder_channels=adjusted_decoder_channels 
        )
        
        # 5. Replace blurry 7x7 conv1 in ResNet-18 with a sharp 3x3 conv1 for satellite parcel edges
        if hasattr(self.model.encoder, "conv1"):
            old_conv = self.model.encoder.conv1
            self.model.encoder.conv1 = nn.Conv2d(
                in_channels=old_conv.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=3,
                stride=old_conv.stride,
                padding=1,
                bias=False
            )

    def forward(self, x):
        """
        Forward pass.
        Input shape: (Batch, Channels, H, W)
        Output shape: (Batch, num_classes, H, W)
        """
        x = self.input_norm(x)
        x = self.spectral_stem(x)
        return self.model(x)