"""
ablation.py

Component-level ablation study for HybridCNNViT: removes ECA, Feature Fusion,
or the Transformer encoder entirely (one at a time) to measure each
component's contribution to final performance.
"""

import torch
import torch.nn as nn
from torchvision import models

from src.models.hybrid_cnn_vit import FeatureFusion, HybridCNNViT
from src.train_utils import evaluate


class HybridNoECA(HybridCNNViT):
    """Ablation variant: skips the ECA attention block."""

    def forward(self, x):
        x = self.backbone(x)
        x = self.project(x)
        x = self.fusion(x)
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.transformer(x)
        # ECA skipped
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class HybridNoFusion(HybridCNNViT):
    """Ablation variant: skips the Feature Fusion block."""

    def forward(self, x):
        x = self.backbone(x)
        x = self.project(x)
        # Fusion skipped
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.transformer(x)
        x = self.eca(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


class HybridNoTransformer(nn.Module):
    """Ablation variant: pools directly after Feature Fusion, no Transformer at all."""

    def __init__(self, embed_dim=192, dropout=0.1, num_classes=2):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.project = nn.Conv2d(512, embed_dim, 1)
        self.fusion = FeatureFusion(embed_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.project(x)
        x = self.fusion(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def load_ablation(cls, checkpoint_path, device, strict=False):
    model = cls().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=strict)  # strict=False: ignores missing keys (e.g. dropped ECA)
    return model.eval()


def run_ablation_study(test_loader, device, hybrid_ckpt="checkpoints/model_hybrid.pt"):
    model_full = HybridCNNViT().to(device)
    model_full.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
    model_full.eval()

    model_no_eca = load_ablation(HybridNoECA, hybrid_ckpt, device)
    model_no_fusion = load_ablation(HybridNoFusion, hybrid_ckpt, device)
    model_no_transformer = HybridNoTransformer().to(device).eval()  # random init: fair "no transformer" baseline

    variants = {
        "Full Hybrid": model_full,
        "No ECA": model_no_eca,
        "No Fusion": model_no_fusion,
        "No Transformer": model_no_transformer,
    }

    results = {name: evaluate(model, test_loader, device) for name, model in variants.items()}

    print(f"\n{'Variant':<20} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 46)
    for name, res in results.items():
        print(f"{name:<20} {res['acc']:>8.4f} {res['f1']:>8.4f} {res['auc']:>8.4f}")

    return results
