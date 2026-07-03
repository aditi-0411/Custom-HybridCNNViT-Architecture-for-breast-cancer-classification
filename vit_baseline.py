"""vit_baseline.py — ImageNet-pretrained ViT-Tiny (timm) baseline for BreakHis."""

import timm


def build_vit_tiny(num_classes=2, pretrained=True):
    return timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=num_classes)
