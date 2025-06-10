import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob

def generate_bcdr_csv(dataset_root, output_csv_path):
    """
    Generate a consolidated CSV from all '*outlines*.csv' files found under dataset_root,
    keeping only specific columns and adding 'path_in_my_folder' if the corresponding
    cropped image exists in the processed dataset folder.
    """
    print("Searching for outline CSVs...")
    dataset_root = Path(dataset_root)
    processed_dataset_root = Path("/home/christian/BCDR/Dataset/BCDR")
    output_rows = []

    required_columns = ['patient_id', 'study_id', 'series', 'lesion_id', 'age', 'density', 'classification']

    for collection in os.listdir(dataset_root):
        collection_path = dataset_root / collection
        if not collection_path.is_dir():
            continue

        csv_files = list(collection_path.glob("*outlines*.csv"))
        if not csv_files:
            print(f"No outlines CSV found in {collection}")
            continue

        for csv_file in csv_files:
            print(f"Processing {csv_file.name}")
            df = pd.read_csv(csv_file)

            if not all(col in df.columns for col in required_columns):
                print(f"Missing required columns in {csv_file.name}. Skipping.")
                continue

            for _, row in tqdm(df.iterrows(), total=len(df)):
                try:
                    patient_id = str(row['patient_id']).strip()
                    study_id = str(row['study_id']).strip()
                    series = str(row['series']).strip()
                    lesion_id = str(row['lesion_id']).strip()

                    # Build pattern for the cropped image
                    search_pattern = f"cropped_img_{patient_id}_{study_id}_{series}_*_mask_id_{lesion_id}.png"

                    # Search recursively under the processed dataset folder
                    matches = glob.glob(str(processed_dataset_root / "**" / search_pattern), recursive=True)

                    if matches:
                        path_in_my_folder = os.path.abspath(matches[0])
                        filtered_row = {col: row[col] for col in required_columns}
                        filtered_row['path_in_my_folder'] = path_in_my_folder
                        output_rows.append(filtered_row)

                except KeyError as e:
                    print(f"Missing column in row: {e}")
                    continue

    if output_rows:
        combined_df = pd.DataFrame(output_rows)
        combined_df.to_csv(output_csv_path, index=False)
        print(f"\nCSV saved at {output_csv_path} with {len(combined_df)} rows (only those with existing cropped images).")
    else:
        print("No valid entries with images found.")
