import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.cbis_ddsm.dataset import ResNetCBISDDSM
from datasets.bcdr.test import evaluate_bcdr_model
from models.resnet_model import BreastCancerResNet18
from transforms.transforms import train_transforms, test_transforms
from datasets.cbis_ddsm.config import TRAIN_CSV as DDSM_TRAIN_CSV, VAL_CSV as DDSM_VAL_CSV, IMG_DIR as DDSM_IMG_DIR
from datasets.bcdr.config import IMG_DIR as BCDR_IMG_DIR, TEST_CSV as BCDR_TEST_CSV
from datasets.cbis_ddsm.config import DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS

PATIENCE = 5

def find_latest_model():
    """
    Find the latest saved best_model.pth from previous runs.
    Returns the path to the best_model.pth file or None.
    """
    results_dir = "results/train_ddsm_test_bcdr"
    if not os.path.exists(results_dir):
        return None

    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not subdirs:
        return None

    # Get the latest directory, which is not the current one
    latest_dir = max(subdirs, key=os.path.getmtime)
    if latest_dir != results_dir:
        print(f"Found previous execution at {latest_dir}")
        best_model_path = os.path.join(latest_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            return best_model_path
    return None

def save_results(train_loss_history, val_loss_history, train_time, trial_dir):
    # Save loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label="Training Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(trial_dir, "loss_curves.png"))
    plt.close()

    # Save training time
    with open(os.path.join(trial_dir, "training_time.txt"), "w") as f:
        f.write(f"Training Time (s): {train_time:.2f}\n")

def run_experiment():
    print("\n[INFO] Starting Training on CBIS-DDSM and Evaluation on BCDR")

    # First, try to find and load the best model from previous runs
    latest_model_path = find_latest_model()

    # Create trial directory ONCE per run (with timestamp)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
    trial_dir = os.path.join("results/train_ddsm_test_bcdr", timestamp)
    os.makedirs(trial_dir, exist_ok=True)

    # Load datasets
    train_dataset = ResNetCBISDDSM(csv_file=DDSM_TRAIN_CSV, root_dir=DDSM_IMG_DIR, transform=train_transforms)
    val_dataset = ResNetCBISDDSM(csv_file=DDSM_VAL_CSV, root_dir=DDSM_IMG_DIR, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize the model
    model = BreastCancerResNet18().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # If a pretrained model exists, load it
    if latest_model_path:
        model.load_state_dict(torch.load(latest_model_path))
        print(f"Loaded pretrained model from {latest_model_path}")
    else:
        print("No pretrained model found. Training from scratch.")

    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    start_time = time.time()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images, labels = batch["image"].to(DEVICE), batch["label"].float().to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch["image"].to(DEVICE), batch["label"].float().to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Time: {epoch_time:.2f} sec")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(trial_dir, "best_model.pth"))
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

    # Final model saving
    total_time = time.time() - start_time
    print(f"[INFO] Training completed in {total_time:.2f} seconds")

    final_model_path = os.path.join(trial_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] Best model saved to: {trial_dir}")

    model.load_state_dict(torch.load(os.path.join(trial_dir, "best_model.pth")))
    evaluate_bcdr_model(model, trial_dir, total_time)
