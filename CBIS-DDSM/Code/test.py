import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from resnet_model import BreastCancerResNet18
from dataset import ResNetCBISDDSM
from transforms import test_transforms
from config import TEST_CSV, IMG_DIR, BATCH_SIZE, DEVICE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
import time
import os
from BreastDensityJSONDataset import BreastDensityJSONDataset


def evaluate_model(model, save_dir, train_time):
    """
    Evaluates the trained model on the standard test dataset (from CSV).
    """
    print("\nStarting Evaluation Process...")

    test_dataset = ResNetCBISDDSM(csv_file=TEST_CSV, root_dir=IMG_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.to(DEVICE)
    model.eval()

    y_true, y_pred, y_probas = [], [], []

    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["image"].to(DEVICE), batch["label"].float().to(DEVICE)
            outputs = model(images).squeeze()

            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)

            y_true.extend(labels.cpu().numpy().astype(int))
            y_pred.extend(predictions)
            y_probas.extend(probabilities)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probas = np.array(y_probas)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_probas)
    except ValueError:
        roc_auc = float("nan")

    test_time = time.time() - start_time

    results_path = os.path.join(save_dir, "accuracy_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Training Time: {train_time:.2f} sec\n")
        f.write(f"Testing Time: {test_time:.2f} sec\n")

    print("\nEvaluation results saved!")

    # ROC Curve
    plt.figure(figsize=(8, 6))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probas)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    except ValueError as e:
        print("Could not plot ROC curve:", e)
        plt.plot([], [], label="ROC unavailable")

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_curve_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC Curve saved at: {roc_curve_path}")

    # Confusion Matrix
    class_names = ["Benign", "Malignant"]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)
    conf_matrix_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Confusion Matrix saved at: {conf_matrix_path}")


def evaluate_model_on_json(model, json_path, save_dir, train_time):
    """
    Evaluates the trained model on a JSON-defined breast density dataset.
    """
    print("\nStarting Evaluation with Breast Density JSON Dataset...")

    test_dataset = BreastDensityJSONDataset(json_path=json_path, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.to(DEVICE)
    model.eval()

    y_true, y_pred, y_probas = [], [], []

    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["image"].to(DEVICE), batch["label"].float().to(DEVICE)
            outputs = model(images).squeeze()

            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)

            y_true.extend(labels.cpu().numpy().astype(int))
            y_pred.extend(predictions)
            y_probas.extend(probabilities)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probas = np.array(y_probas)

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_probas)
    except ValueError:
        roc_auc = float("nan")

    test_time = time.time() - start_time

    results_path = os.path.join(save_dir, "accuracy_results.json_test.txt")
    with open(results_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Training Time: {train_time:.2f} sec\n")
        f.write(f"Testing Time: {test_time:.2f} sec\n")

    print("JSON Test Evaluation results saved!")

    # ROC Curve
    plt.figure(figsize=(8, 6))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probas)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    except ValueError as e:
        print("Could not plot ROC curve:", e)
        plt.plot([], [], label="ROC unavailable")

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (JSON Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_curve_path = os.path.join(save_dir, "roc_curve_json_test.png")
    plt.savefig(roc_curve_path)
    plt.close()
    print(f"ROC Curve saved at: {roc_curve_path}")

    # Confusion Matrix
    class_names = ["Benign", "Malignant"]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (JSON Test Set)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=30)
    plt.yticks(rotation=30)
    conf_matrix_path = os.path.join(save_dir, "confusion_matrix_json_test.png")
    plt.savefig(conf_matrix_path)
    plt.close()
    print(f"Confusion Matrix saved at: {conf_matrix_path}")
