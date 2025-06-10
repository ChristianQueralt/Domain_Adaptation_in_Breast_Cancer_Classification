from pathlib import Path
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

def extract_crops_from_bcdr(root_path, output_path):
    """
    Traverses the BCDR dataset, finds image-mask pairs, applies cropping, and saves the crops.

    Args:
        root_path (str or Path): Path to the BCDR base directory containing collections.
        output_path (str or Path): Path where the cropped images will be saved.
    """

    root_path = Path(root_path)
    output_path = Path(output_path)

    # Traverse collections
    collections = [d for d in root_path.iterdir() if d.is_dir()]
    for collection in tqdm(collections, desc="Collections"):
        # Traverse patients
        patients = [d for d in collection.iterdir() if d.is_dir() and d.name.startswith('patient')]
        for patient in tqdm(patients, desc=f"Patients in {collection.name}", leave=False):
            # Traverse studies
            studies = [d for d in patient.iterdir() if d.is_dir()]
            for study in studies:
                image_files = list(study.glob('*.tif'))
                for img_file in image_files:
                    # Only process images that have a corresponding mask
                    if "mask_id" not in img_file.name:
                        base_name = img_file.stem
                        mask_candidates = list(study.glob(f"{base_name}_mask_id_*.tif"))
                        if mask_candidates:
                            mask_file = mask_candidates[0]  # Take the first mask if multiple

                            # Load image and mask
                            image = Image.open(img_file).convert('L')
                            mask = Image.open(mask_file).convert('L')

                            # Find bounding box of the mask
                            mask_array = np.array(mask)
                            non_zero = np.argwhere(mask_array > 0)

                            if non_zero.size == 0:
                                print(f"Warning: Empty mask for {mask_file}")
                                continue

                            y_min, x_min = non_zero.min(axis=0)
                            y_max, x_max = non_zero.max(axis=0)

                            # Crop the image around the mask
                            cropped = image.crop((x_min, y_min, x_max, y_max))

                            # Build output path
                            relative_path = img_file.relative_to(root_path)
                            save_path = output_path / relative_path.parent
                            save_path.mkdir(parents=True, exist_ok=True)

                            cropped_filename = f"cropped_{mask_file.stem}.png"
                            cropped.save(save_path / cropped_filename)

                            print(f"Saved crop: {save_path / cropped_filename}")