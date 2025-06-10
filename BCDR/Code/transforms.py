import torch
from torchvision.transforms import v2
import torchvision.transforms.functional as F
import random

# Define transformations for training data (includes augmentations)
train_transforms = v2.Compose([
    v2.ToImage(),  # Ensure input is a tensor image
    v2.Resize((128, 128), antialias=True),
    #v2.Resize((224, 224), antialias=True),# Resize images to 224x224

    # Geometric transformations
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=(90,90)),

    v2.ToDtype(torch.float32, scale=True),  # Normalize to [0, 1]

    # Randomly apply one color transformation
    v2.RandomChoice([
        v2.Lambda(lambda img: F.adjust_brightness(img, brightness_factor=1.25)), #adjusted
        v2.Lambda(lambda img: F.adjust_contrast(img, contrast_factor=1.25)), #adjusted
        v2.Lambda(lambda img: F.adjust_gamma(img, gamma=1.25, gain=1.25)), #adjusted
        v2.Lambda(lambda img: F.adjust_saturation(img, saturation_factor=3)), #adjusted
        v2.Lambda(lambda img: F.adjust_sharpness(img, sharpness_factor=6)), #adjusted
        v2.Lambda(lambda img: F.autocontrast(img)),
        v2.Lambda(lambda img: F.invert(img)),
        v2.GaussianBlur(kernel_size=11, sigma=5.5),  # Blur image
        v2.GaussianNoise(mean=0.0, sigma=random.uniform(0.05, 0.1), clip=True),  # Adjusted
        v2.Lambda(lambda img: img),
    ], p = [0.11875, 0.11875, 0.11875, 0.11875, 0.11875, 0.025, 0.025, 0.11875, 0.11875, 0.11875]),

    #p = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
])

# Define transformations for validation and test data (no augmentation)
test_transforms = v2.Compose([
    v2.ToImage(),
    #v2.Resize((224, 224), antialias=True),# Resize images to 224x224
    v2.Resize((128, 128), antialias=True),

    v2.ToDtype(torch.float32, scale=True),
])
