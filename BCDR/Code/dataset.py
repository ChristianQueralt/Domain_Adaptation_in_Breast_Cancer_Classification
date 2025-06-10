import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

class BCDRDataset(Dataset):
    """
    BCDR Dataset for Breast Cancer Binary Classification.
    Labels:
    0 -> Benign
    1 -> Malign
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations and paths.
            root_dir (string): Directory containing the images.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.csv = csv_file  # Store CSV file path
        self.annotations = pd.read_csv(csv_file, sep=',')  # Load CSV with comma separator
        self.root_dir = root_dir  # Root directory where images are stored
        self.transform = transform  # Image transformations

        print(f"BCDRDataset initialized with {len(self.annotations)} samples from {self.csv}.")

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Get a single sample (image and label) by index.

        Returns:
            dict: A dictionary with the image tensor and the binary label tensor.
        """
        row = self.annotations.iloc[idx]

        # Construct full image path
        image_path = os.path.normpath(os.path.join(self.root_dir, row['path_in_my_folder']))

        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Load image in RGB format
        image = Image.open(image_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if not isinstance(image, torch.Tensor):
            image = v2.ToImage()(image)

        # Map "Malign" → 1, "Benign" → 0
        classification = str(row['classification']).strip().upper()
        label = 1 if classification == "MALIGN" else 0
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return {"image": image, "label": label_tensor}
