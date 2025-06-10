import torch
import torchvision
import matplotlib.pyplot as plt
from dataset import ResNetCBISDDSM
from transforms import train_transforms, test_transforms
from config import TRAIN_CSV, VAL_CSV, TEST_CSV, IMG_DIR, BATCH_SIZE

from torch.utils.data import DataLoader

# Load datasets
train_dataset = ResNetCBISDDSM(csv_file=TRAIN_CSV, root_dir=IMG_DIR, transform=train_transforms)
val_dataset = ResNetCBISDDSM(csv_file=VAL_CSV, root_dir=IMG_DIR, transform=test_transforms)
test_dataset = ResNetCBISDDSM(csv_file=TEST_CSV, root_dir=IMG_DIR, transform=test_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

def show_images(loader, title="Batch Images"):
    """
    Function to visualize a batch of images from a DataLoader.

    Args:
        loader (DataLoader): The DataLoader containing the images.
        title (str): Title of the plot.
    """
    images = next(iter(loader))["image"]  # Get a batch of images
    grid_img = torchvision.utils.make_grid(images, nrow=4, normalize=True)  # Create grid of images

    # Convert tensor to NumPy format and plot
    plt.figure(figsize=(10, 5))
    plt.imshow(grid_img.permute(1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
    plt.title(title)
    plt.axis("off")
    plt.show()


# Example usage
if __name__ == "__main__":
    print("Training Images:")
    show_images(train_loader, title="Training Batch Images")

    print("\nValidation Images:")
    show_images(val_loader, title="Validation Batch Images")

    print("\nTest Images:")
    show_images(test_loader, title="Test Batch Images")
