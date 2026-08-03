import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    """
    Wrapper class for PyTorch's ResNet-18, customized for 
    multispectral/hyperspectral image patches.
    """
    def __init__(self, in_channels=10, num_classes=4, use_gpu=True):
        super(ResNet18, self).__init__()
        
        # Load the base ResNet18 architecture
        self.model = resnet18(weights=None)
        self.name = "ResNet-18"
        
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
            self.cuda()  # Moves the entire wrapper and submodules to GPU
            print("[ResNet18] Initialized and moved to GPU.")
        else:
            print("[ResNet18] Initialized on CPU.")

    def forward(self, x):
        return self.model(x)