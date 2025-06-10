# experiments/transfer_learning_BCDR_percentages_shade.py

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

def evaluate_auc(model, val_loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(DEVICE)
            l = batch["label"].float().to(DEVICE)
            out = torch.sigmoid(model(images))
            preds.extend(out.cpu().numpy())
            labels.extend(l.cpu().numpy())
    return roc_auc_score(labels, preds)

def train_and_eval(pct, pretrain_loader, bcdr_dataset, val_loader):
    model = BreastCancerResNet18().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Pretraining on full DDSM with early stopping
    best_loss, patience_counter = float('inf'), 0
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
        if avg_loss < best_loss:
            best_loss, patience_counter = avg_loss, 0
        else:
            patience_counter += 1
        if patience_counter >= 10:
            break

    # Fine-tuning on BCDR subset
    n_samples = int((pct / 100.0) * len(bcdr_dataset))
    bcdr_subset = Subset(bcdr_dataset, list(range(n_samples)))
    fine_tune_loader = DataLoader(bcdr_subset, batch_size=BATCH_SIZE, shuffle=True)

    best_loss, patience_counter = float('inf'), 0
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
        if avg_loss < best_loss:
            best_loss, patience_counter = avg_loss, 0
        else:
            patience_counter += 1
        if patience_counter >= 10:
            break

    return evaluate_auc(model, val_loader)

def run_experiment():
    torch.manual_seed(65)
    random.seed(65)
    np.random.seed(65)

    os.makedirs('results', exist_ok=True)

    ddsm_dataset = ResNetCBISDDSM(csv_file=DDSM_TRAIN_CSV, root_dir=DDSM_IMG_DIR, transform=train_transforms)
    pretrain_loader = DataLoader(ddsm_dataset, batch_size=BATCH_SIZE, shuffle=True)

    bcdr_dataset = BCDRDataset(csv_file=BCDR_TRAIN_CSV, root_dir=BCDR_IMG_DIR, transform=train_transforms)
    val_dataset = BCDRDataset(csv_file=BCDR_VAL_CSV, root_dir=BCDR_IMG_DIR, transform=test_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    percentages = list(range(5, 105, 5))
    all_runs = []

    print("\n>>> Running 4 iterations for shaded plot")
    for run in range(4):
        run_aucs = []
        print(f"\n--- Run {run+1} ---")
        for pct in percentages:
            print(f"Fine-tuning with {pct}% BCDR")
            auc = train_and_eval(pct, pretrain_loader, bcdr_dataset, val_loader)
            run_aucs.append(auc)
            print(f"AUC: {auc:.4f}")
        all_runs.append(run_aucs)

    # Compute mean, min, max AUC for shaded plot
    all_runs = np.array(all_runs)
    mean_auc = np.mean(all_runs, axis=0)
    min_auc = np.min(all_runs, axis=0)
    max_auc = np.max(all_runs, axis=0)

    # Train from scratch with 100% BCDR
    print("\nTraining baseline model from scratch with 100% BCDR")
    model = BreastCancerResNet18().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    full_loader = DataLoader(bcdr_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_loss, patience_counter = float('inf'), 0
    for epoch in range(100):
        model.train()
        losses = []
        for batch in full_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].float().to(DEVICE)
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

    auc_scratch = evaluate_auc(model, val_loader)
    print(f"Baseline AUC (scratch): {auc_scratch:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, mean_auc, marker='o', color='blue', label='Mean AUC (4 runs)')
    plt.fill_between(percentages, min_auc, max_auc, color='blue', alpha=0.2, label='Min-Max Range')
    plt.axhline(y=auc_scratch, color='green', linestyle='--', label='BCDR Scratch AUC')
    plt.xlabel('Percentage of BCDR used for fine-tuning')
    plt.ylabel('AUC on BCDR Validation Set')
    plt.title('Transfer Learning Performance with Shaded Variability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/transfer_learning_4runs_shaded_100_epochs.png')
    plt.show()
