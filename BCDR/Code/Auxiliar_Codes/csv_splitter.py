import pandas as pd
import os
import argparse

def split_csv_by_patient(csv_path, train_ratio):
    """
    Splits a CSV file into training and testing sets based on unique patient_id,
    ensuring no patient is shared between both sets.

    Parameters:
    - csv_path (str): Path to the original CSV file.
    - train_ratio (float): Percentage of data to allocate to training (e.g., 0.8 for 80%).

    Outputs:
    - Two new CSV files: *_train.csv and *_test.csv in the same directory as the input.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Group by patient_id to avoid splitting a patient across sets
    patient_ids = df['patient_id'].unique()
    total_patients = len(patient_ids)
    print(f"Total unique patients: {total_patients}")

    # Shuffle patients
    patient_ids = pd.Series(patient_ids).sample(frac=1, random_state=42).tolist()

    # Determine split index based on cumulative patient allocation
    split_index = int(total_patients * train_ratio)
    train_patients = set(patient_ids[:split_index])
    test_patients = set(patient_ids[split_index:])

    # Select rows
    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    test_df = df[df['patient_id'].isin(test_patients)].reset_index(drop=True)

    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Save to new CSV files
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.dirname(csv_path)
    train_path = os.path.join(output_dir, f"{base_name}_train.csv")
    test_path = os.path.join(output_dir, f"{base_name}_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nFiles saved:\n  - {train_path}\n  - {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CSV by patient without overlap")
    parser.add_argument("csv_path", type=str, help="Path to the original CSV")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train set ratio (default: 0.8)")
    args = parser.parse_args()

    split_csv_by_patient(args.csv_path, args.train_ratio)
