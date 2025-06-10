import torch
import torch.nn as nn
import torchvision.models as models

class BreastCancerResNet18(nn.Module):
    """
    Custom ResNet18 for binary breast cancer classification (Benign vs Malignant).
    """

    def __init__(self, pretrained=True, num_classes=1):
        super(BreastCancerResNet18, self).__init__()

        # Load pre-trained ResNet18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the FC layer with a new head for binary classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Output layer for binary classification
        )

        # Unfreeze new FC layers
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
