"""resnet_baseline.py — ImageNet-pretrained ResNet18 baseline for BreakHis."""

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes=2):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
