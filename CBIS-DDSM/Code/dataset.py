import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

class ResNetCBISDDSM(Dataset):
    """
    CBIS-DDSM Dataset for Breast Cancer Image Classification.
    Now classifies into 2 categories:
    0 -> Malignant
    1 -> Benign
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations and paths.
            root_dir (string): Directory containing the images.
            transform (callable, optional): Optional transform to be applied
                                            on a sample (e.g., image normalization).
        """
        self.csv = csv_file  # Store CSV file path
        self.annotations = pd.read_csv(csv_file, sep=';')  # Load CSV as DataFrame
        self.root_dir = root_dir  # Root directory where images are stored
        self.transform = transform  # Optional transformations

        print(f"Dataset initialized with {len(self.annotations)} samples from {self.csv}.")

    def __len__(self):
        """ Returns the number of samples in the dataset. """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Fetch a single sample (image and label) from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            dict: A dictionary containing the processed image tensor and its corresponding label.
        """
        row = self.annotations.iloc[idx]

        # Get image path and label
        image_path = os.path.normpath(os.path.join(self.root_dir, row['updated png cropped image file path']))
        pathology = row['pathology']
        abnormality = row['abnormality type']

        """
        4 LABEL VERSION
        
        # Determine label based on pathology and abnormality type
        if pathology in ['MALIGNANT']:
            label = 0 if abnormality.lower() == "calcification" else 1  # Malignant CALCIFICATION (0) or Malignant MASS (1)
        else:  # BENIGN or BENIGN_WITHOUT_CALLBACK
            label = 2 if abnormality.lower() == "calcification" else 3  # Benign CALCIFICATION (2) or Benign MASS (3)
        """


        label = 1 if pathology.strip().upper() == "MALIGNANT" else 0

        # Check if the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Load image with PIL and ensure it has 3 channels (RGB)
        image = Image.open(image_path).convert("RGB")

        # Apply transformations (if defined)
        if self.transform:
            image = self.transform(image)  # Transform function expects a PIL image

        # Ensure the image is a tensor before returning
        if not isinstance(image, torch.Tensor):
            image = v2.ToImage()(image)  # Convert to tensor if it's not already

        # Convert label to tensor
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return {'image': image, 'label': label_tensor}
