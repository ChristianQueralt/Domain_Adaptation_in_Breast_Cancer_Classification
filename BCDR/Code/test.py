import os
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    roc_curve, auc, confusion_matrix
)
from torch.utils.data import DataLoader
from dataset import BCDRDataset
from transforms import test_transforms
from config import TEST_CSV, IMG_DIR, BATCH_SIZE, DEVICE
from resnet_model import BreastCancerResNet18

def evaluate_bcdr_model(model, save_dir, train_time):
    print("\nEvaluating BCDR model...")

    # Load dataset
    test_dataset = BCDRDataset(csv_file=TEST_CSV, root_dir= IMG_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Evaluation mode
    model.to(DEVICE)
    model.eval()

    y_true, y_pred, y_probas = [], [], []

    start = time.time()

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
            preds = (probs > 0.5).astype(int)

            y_true.extend(labels.cpu().numpy().squeeze())
            y_pred.extend(preds)
            y_probas.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probas = np.array(y_probas)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_probas)
    test_duration = time.time() - start

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "bcdr_results.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
        f.write(f"ROC AUC: {roc:.4f}\n")
        f.write(f"Training Time: {train_time:.2f} sec\n")
        f.write(f"Testing Time: {test_duration:.2f} sec\n")

    print("Results saved.")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_probas)
    auc_score = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign", "Malign"], yticklabels=["Benign", "Malign"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    print(f"ROC and Confusion Matrix saved in {save_dir}")
