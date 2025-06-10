# experiments/transfer_learning_BCDR_percentages.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import random

from datasets.bcdr.dataset import BCDRDataset
from datasets.cbis_ddsm.dataset import ResNetCBISDDSM
from models.resnet_model import BreastCancerResNet18
from transforms.transforms import train_transforms, test_transforms
from datasets.bcdr.config import TRAIN_CSV as BCDR_TRAIN_CSV, VAL_CSV as BCDR_VAL_CSV, IMG_DIR as BCDR_IMG_DIR
from datasets.cbis_ddsm.config import TRAIN_CSV as DDSM_TRAIN_CSV, IMG_DIR as DDSM_IMG_DIR
from datasets.cbis_ddsm.config import DEVICE, BATCH_SIZE, LEARNING_RATE

def run_experiment():
    torch.manual_seed(65)
    random.seed(65)
    np.random.seed(65)

    # Load full DDSM dataset for pretraining
    ddsm_dataset = ResNetCBISDDSM(csv_file=DDSM_TRAIN_CSV, root_dir=DDSM_IMG_DIR, transform=train_transforms)
    pretrain_loader = DataLoader(ddsm_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load BCDR dataset
    bcdr_dataset = BCDRDataset(csv_file=BCDR_TRAIN_CSV, root_dir=BCDR_IMG_DIR, transform=train_transforms)
    val_dataset = BCDRDataset(csv_file=BCDR_VAL_CSV, root_dir=BCDR_IMG_DIR, transform=test_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    percentages = list(range(5, 105, 5))
    auc_scores = []

    for pct in percentages:
        print(f"\n>>> Fine-tuning with {pct}% of BCDR dataset")

        # Initialize model and optimizer
        model = BreastCancerResNet18().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        # Pretraining on full DDSM (early stopping enabled)
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        for epoch in range(100):
            model.train()
            running_loss = 0.0
            for batch in pretrain_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].float().to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(pretrain_loader)
            print(f"[Pretrain] Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("[Pretrain] Early stopping")
                break

        # Fine-tuning on partial BCDR
        n_samples = int((pct / 100.0) * len(bcdr_dataset))
        bcdr_subset = Subset(bcdr_dataset, list(range(n_samples)))
        fine_tune_loader = DataLoader(bcdr_subset, batch_size=BATCH_SIZE, shuffle=True)

        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(100):
            model.train()
            running_loss = 0.0
            for batch in fine_tune_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].float().to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(fine_tune_loader)
            print(f"[Fine-tune {pct}%] Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"[Fine-tune {pct}%] Early stopping")
                break

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                labels = batch["label"].float().to(DEVICE)
                outputs = torch.sigmoid(model(images))
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        auc = roc_auc_score(all_labels, all_preds)
        auc_scores.append(auc)
        print(f"AUC with {pct}% BCDR: {auc:.4f}")

    # Train model from scratch with 100% BCDR
    print("\nTraining from scratch with full BCDR dataset")
    model = BreastCancerResNet18().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    full_loader = DataLoader(bcdr_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_loss, patience_counter = float('inf'), 0
    for epoch in range(100):
        model.train()
        losses = []
        for batch in full_loader:
            images, labels = batch["image"].to(DEVICE), batch["label"].float().to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        if avg_loss < best_loss:
            best_loss, patience_counter = avg_loss, 0
        else:
            patience_counter += 1
        if patience_counter >= 10:
            break

    # Evaluate scratch model
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            l = batch["label"].float().to(DEVICE)
            out = torch.sigmoid(model(images))
            preds.extend(out.cpu().numpy())
            labels.extend(l.cpu().numpy())
    auc_scratch = roc_auc_score(labels, preds)
    print(f"AUC from scratch with full BCDR: {auc_scratch:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, auc_scores, marker='o', label='Transfer Learning AUC')
    plt.axhline(y=max(auc_scores), color='r', linestyle='--', label='Best TL AUC')
    plt.axhline(y=auc_scratch, color='g', linestyle=':', label='BCDR Scratch AUC')
    plt.xlabel('Percentage of BCDR used for fine-tuning')
    plt.ylabel('AUC on BCDR Validation Set')
    plt.title('Transfer Learning vs. Full BCDR Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/transfer_learning_vs_full.png')
    plt.show()
