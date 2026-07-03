"""
compare_models.py

Loads all three trained checkpoints (ResNet18, ViT-Tiny, HybridCNNViT) and
evaluates them side by side on the held-out test set.

NOTE: the original notebook had a bug here — it instantiated ViT-Tiny with
`pretrained=False` and never called `load_state_dict` on the fine-tuned
checkpoint before evaluating, so the comparison table silently scored a
randomly-initialized ViT-Tiny. This version loads all three checkpoints
correctly.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.models import HybridCNNViT, build_vit_tiny
from src.train_utils import evaluate


def load_resnet18(checkpoint_path, device, num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()


def load_vit_tiny(checkpoint_path, device, num_classes=2):
    model = build_vit_tiny(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()


def load_hybrid(checkpoint_path, device):
    model = HybridCNNViT()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()


def compare(test_loader, device, resnet_ckpt, vit_ckpt, hybrid_ckpt):
    model_cnn = load_resnet18(resnet_ckpt, device)
    model_vit = load_vit_tiny(vit_ckpt, device)
    model_hybrid = load_hybrid(hybrid_ckpt, device)

    results = {
        "ResNet18": evaluate(model_cnn, test_loader, device),
        "ViT-Tiny": evaluate(model_vit, test_loader, device),
        "Hybrid": evaluate(model_hybrid, test_loader, device),
    }

    metrics = ["acc", "precision", "recall", "f1", "auc"]
    print(f"\n{'Metric':<12} {'ResNet18':>10} {'ViT-Tiny':>10} {'Hybrid':>10}")
    print("-" * 44)
    for m in metrics:
        print(f"{m:<12} {results['ResNet18'][m]:>10.4f} {results['ViT-Tiny'][m]:>10.4f} {results['Hybrid'][m]:>10.4f}")

    return results


if __name__ == "__main__":
    from src.dataset import get_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders()

    compare(
        test_loader,
        device,
        resnet_ckpt="checkpoints/model_cnn.pt",
        vit_ckpt="checkpoints/model_vit.pt",
        hybrid_ckpt="checkpoints/model_hybrid.pt",
    )
