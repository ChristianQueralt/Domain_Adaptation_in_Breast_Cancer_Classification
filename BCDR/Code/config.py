import torch

# Device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths
TRAIN_CSV = "/home/christian/BCDR/Dataset/updated_train_split_case_description_train_set.csv"
VAL_CSV = "/home/christian/BCDR/Dataset/updated_validation_split_case_description_train_set.csv"
TEST_CSV = "/home/christian/BCDR/Dataset/updated_case_description_test_set.csv"

# Root dir is not used here because paths are absolute in the CSVs
IMG_DIR = "/home/christian/BCDR/Dataset/BCDR"  # for compatibility only

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100
