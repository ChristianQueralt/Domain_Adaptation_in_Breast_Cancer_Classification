import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from dataset import BCDRDataset
from resnet_model import BreastCancerResNet18
from transforms import train_transforms, test_transforms
from config import TRAIN_CSV, VAL_CSV, IMG_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, DEVICE
from test import evaluate_bcdr_model
import os
import datetime
import matplotlib.pyplot as plt

PATIENCE = 5

def find_latest_model():
    """
    Search for the latest saved best_model.pth in /home/christian/BCDR/Results.
    Returns the path to the best_model.pth file or None.
    """
    results_dir = "/home/christian/BCDR/Results"
    if not os.path.exists(results_dir):
        return None

    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not subdirs:
        return None

    latest_dir = max(subdirs, key=os.path.getmtime)
    best_model_path = os.path.join(latest_dir, "best_model.pth")
    return best_model_path if os.path.exists(best_model_path) else None

def save_results(train_loss_history, val_loss_history, train_time, trial_dir=None):
    results_dir = "/home/christian/BCDR/Results"
    os.makedirs(results_dir, exist_ok=True)

    if trial_dir is None:
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
        trial_dir = os.path.join(results_dir, timestamp)
        os.makedirs(trial_dir, exist_ok=True)

    if train_loss_history and val_loss_history:
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

    return trial_dir

def train_model():
    print("\nStarting Training on BCDR Dataset...")

    train_dataset = BCDRDataset(csv_file=TRAIN_CSV, root_dir=IMG_DIR, transform=train_transforms)
    val_dataset = BCDRDataset(csv_file=VAL_CSV, root_dir=IMG_DIR, transform=test_transforms)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BreastCancerResNet18().to(DEVICE)

    # Load previous best model if available
    latest_model_path = find_latest_model()
    if latest_model_path:
        model.load_state_dict(torch.load(latest_model_path))
        print(f"Loaded pretrained model from {latest_model_path}")
    else:
        print("No pretrained model found. Training from scratch.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    trial_dir = save_results([], [], 0.0)

    best_val_loss = float('inf')
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].float().to(DEVICE).unsqueeze(1)
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
                images = batch["image"].to(DEVICE)
                labels = batch["label"].float().to(DEVICE).unsqueeze(1)
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
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    torch.save(model.state_dict(), os.path.join(trial_dir, "final_model.pth"))
    save_results(train_loss_history, val_loss_history, total_time, trial_dir)
    evaluate_bcdr_model(model, trial_dir, total_time)
