import torch
import torch.nn as nn
import torchvision.models as models

class BreastCancerResNet18(nn.Module):
    """
    Custom ResNet18 for breast cancer classification with partial backbone freezing.
    """

    def __init__(self, pretrained=True):
        super(BreastCancerResNet18, self).__init__()

        # Load ResNet18 with pre-trained weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Freeze all layers initially
        for param in self.resnet.parameters():
            param.requires_grad = False

        """# Unfreeze the deeper layers (layer4 and fc)
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True"""

        # Get the number of features from the original FC layer
        num_features = self.resnet.fc.in_features

        # Replace the FC head with custom classification layers and dropout
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        """self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Single output for binary classification
        )"""

        # Unfreeze the new FC layers
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
