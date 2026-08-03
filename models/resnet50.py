import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50(nn.Module):
    """
    Wrapper class for PyTorch's ResNet-50, customized for 
    multispectral/hyperspectral image patches.
    """
    def __init__(self, in_channels=10, num_classes=4, use_gpu=True):
        super(ResNet50, self).__init__()
        
        # Load the base ResNet50 architecture
        self.model = resnet50(weights=None)
        
        # Modify the first convolutional layer
        self.model.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Modify the final Fully Connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # GPU configuration
        self.use_gpu = use_gpu
        if self.use_gpu and torch.cuda.is_available():
            self.cuda()
            print("[ResNet50] Initialized and moved to GPU.")
        else:
            print("[ResNet50] Initialized on CPU.")

    def forward(self, x):
        return self.model(x)