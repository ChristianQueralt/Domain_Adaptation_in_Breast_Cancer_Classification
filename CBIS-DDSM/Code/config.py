import torch

# Define the device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths to CSV files
TRAIN_CSV = r"/home/christian/Dataset/MERGED_CSVs/updated_train_split_merged_case_description_train_set.csv"
VAL_CSV = r"/home/christian/Dataset/MERGED_CSVs/updated_validation_split_merged_case_description_train_set.csv"
TEST_CSV = r"/home/christian/Dataset/MERGED_CSVs/updated_merged_case_description_test_set.csv"

# Directory containing images
IMG_DIR = r"/home/christian/CBIS-DDSM/Dataset/CBIS-DDSM"

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
NUM_EPOCHS = 200
