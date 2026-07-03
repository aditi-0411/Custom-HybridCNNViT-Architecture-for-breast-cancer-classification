"""
train_utils.py

Shared training loop, evaluation function, and class-weighted loss
construction used across all three models (ResNet18, ViT-Tiny, HybridCNNViT).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def class_weighted_criterion(train_loader, device):
    """Builds a CrossEntropyLoss weighted inversely by class frequency."""
    labels = []
    for _, y in train_loader:
        labels.extend(y.cpu().numpy().tolist())

    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum()

    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, train=False):
    """Runs one epoch of training or evaluation and returns a metrics dict."""
    model.train() if train else model.eval()

    losses, y_true, y_pred, y_score = [], [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        losses.append(loss.item())
        probs = torch.softmax(outputs, dim=1)[:, 1]
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.argmax(1).cpu().numpy())
        y_score.extend(probs.detach().cpu().numpy())

    return {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
        "y_true": y_true,
        "y_score": y_score,
    }


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, freeze_backbone_params=None):
    """Generic training loop with AdamW + cosine annealing + AMP.

    `freeze_backbone_params`: optional iterable of params to train instead of
    all model params (useful for a frozen-backbone warmup stage).
    """
    criterion = class_weighted_criterion(train_loader, device)
    params = freeze_backbone_params if freeze_backbone_params is not None else model.parameters()
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history = {"train": [], "val": []}
    best_auc, best_state = -1, None

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, scaler, train=True)
        val_metrics = run_epoch(model, val_loader, device, criterion, train=False)
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch:02d} | Train F1 {train_metrics['f1']:.4f} AUC {train_metrics['auc']:.4f} "
            f"| Val F1 {val_metrics['f1']:.4f} AUC {val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_auc


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluates a model on a loader, no gradient / no training-mode side effects."""
    model.eval()
    y_true, y_pred, y_score = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        prob = torch.softmax(out, 1)[:, 1]
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_score.extend(prob.cpu().numpy())

    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
        "y_true": y_true,
        "y_score": y_score,
    }
