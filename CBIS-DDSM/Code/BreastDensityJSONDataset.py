import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import os

class BreastDensityJSONDataset(Dataset):
    """
    Dataset class to load breast density classification images and labels from a JSON file.

    JSON format expected:
    {
      "Test": [
        {"image": "path/to/image.jpg", "label": [1, 0, 0, 0]},
        ...
      ]
    }

    Converts one-hot encoded labels into integer class indices:
    A → 0, B → 1, C → 2, D → 3
    """

    def __init__(self, json_path, transform=None, group_key="Test"):
        """
        Args:
            json_path (str): Path to the JSON file generated with create_dataset.py
            transform (callable, optional): Optional transform to apply to each image
            group_key (str): JSON key to load, default is "Test"
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        self.samples = data[group_key]
        self.transform = transform

        print(f"BreastDensityJSONDataset initialized with {len(self.samples)} images from group '{group_key}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get image path and one-hot label
        image_path = self.samples[idx]["image"]
        label_onehot = self.samples[idx]["label"]

        # Load the image and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Apply image transforms (if any)
        if self.transform:
            image = self.transform(image)

        # Convert one-hot label to class index
        label = torch.tensor(label_onehot).argmax().long()

        return {"image": image, "label": label}
