import time
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from resnet_model import BreastCancerResNet18
from dataset import ResNetCBISDDSM
from transforms import train_transforms, test_transforms
from config import TRAIN_CSV, VAL_CSV, IMG_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DEVICE
from BreastDensityJSONDataset import BreastDensityJSONDataset
from test import evaluate_model, evaluate_model_on_json
import os
import datetime
import matplotlib.pyplot as plt

# Early stopping parameters
PATIENCE = 15
json_path = r"/home/christian/Breast_Density_Classification/breast_density_classification/scripts/configs/my_dataset.json"

def find_latest_model():
    """
    Find the latest saved best_model.pth in /home/christian/Resoults.
    Returns the path to the best_model.pth file.
    """
    results_dir = "/home/christian/CBIS-DDSM/Resoults"
    if not os.path.exists(results_dir):
        return None

    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not subdirs:
        return None

    latest_dir = max(subdirs, key=os.path.getmtime)
    best_model_path = os.path.join(latest_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        return best_model_path
    else:
        return None

def train_model():
    """
    Function to fine-tune the ResNet18 model for breast density classification.
    """

    # Load datasets
    train_dataset = ResNetCBISDDSM(csv_file=TRAIN_CSV, root_dir=IMG_DIR, transform=train_transforms)
    val_dataset = ResNetCBISDDSM(csv_file=VAL_CSV, root_dir=IMG_DIR, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Display sample images
    train_images = next(iter(train_loader))["image"]
    for ig in train_images:
        print(ig.min(), ig.max(), ig.shape)

    grid_img = torchvision.utils.make_grid(train_images, nrow=4, normalize=True)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(r"/home/christian/CBIS-DDSM/Resoults/TEST/Last_Test.png")

    # Initialize model
    model = BreastCancerResNet18(pretrained=False).to(DEVICE)

    # Load latest pretrained weights
    latest_model_path = find_latest_model()

    if latest_model_path:
        model.load_state_dict(torch.load(latest_model_path))
        print(f"Loaded pretrained model from {latest_model_path}")
    else:
        print("No pretrained model found. Training from scratch.")

    # Define class weights (optional, if needed for fine-tuning)
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float).to(DEVICE)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    trial_dir = save_results([], [], 0.0)

    train_loss_history = []
    val_loss_history = []

    best_val_loss = float('inf')
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_loss_history.append(epoch_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch {epoch}/{NUM_EPOCHS} - Training Loss: {epoch_loss:.4f} - Validation Loss: {val_loss:.4f} - Time: {epoch_duration:.2f} sec")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(trial_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # Save final model
    final_model_path = os.path.join(trial_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Training summary
    total_training_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
    print("\nTotal Training Time:", total_training_time / 3600, "hours")
    print(f"Best model saved at: {best_model_path}")
    print(f"Final model saved at: {final_model_path}")

    # Save plots and evaluate
    save_results(train_loss_history, val_loss_history, total_training_time, trial_dir)


    ###### Comment one evaluation model
    evaluate_model(model, trial_dir, total_training_time)
    #evaluate_model_on_json(model,json_path, trial_dir, total_training_time)

def save_results(train_loss_history, val_loss_history, train_time, trial_dir=None):
    results_dir = "/home/christian/CBIS-DDSM/Resoults"
    os.makedirs(results_dir, exist_ok=True)

    if trial_dir is None:
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
        trial_dir = os.path.join(results_dir, timestamp)
        os.makedirs(trial_dir, exist_ok=True)

    # Save graph
    if train_loss_history and val_loss_history:
        graph_path = os.path.join(trial_dir, "graph.png")
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Training Loss")
        plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(graph_path)
        plt.close()

    # Save training config
    settings_path = os.path.join(trial_dir, "settings.txt")
    with open(settings_path, "w") as f:
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"PATIENCE: {PATIENCE}\n")

    print(f"Results saved in: {trial_dir}")
    return trial_dir

if __name__ == "__main__":
    train_model()
