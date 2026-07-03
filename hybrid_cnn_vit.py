"""
hybrid_cnn_vit.py

Core contribution of this project: a lightweight Hybrid CNN-ViT architecture
combining a ResNet18 backbone, a depthwise Feature Fusion block, a shallow
Transformer encoder, and Efficient Channel Attention (ECA).

Architecture:
    Input -> ResNet18 backbone -> 1x1 projection -> FeatureFusion
          -> Conv tokenizer -> +pos_embed -> TransformerEncoder x depth
          -> ECA -> LayerNorm -> global average pool -> MLP head
"""

import torch
import torch.nn as nn
from torchvision import models


class ECABlock(nn.Module):
    """Efficient Channel Attention: near-parameter-free channel re-weighting
    via a 1D conv over the pooled token dimension."""

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, N, C)
        y = x.transpose(1, 2)
        y = self.pool(y)
        y = y.transpose(1, 2)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


class FeatureFusion(nn.Module):
    """Depthwise-separable residual block that refines CNN feature maps
    before they are tokenized for the Transformer."""

    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x + identity


class HybridCNNViT(nn.Module):
    def __init__(self, embed_dim=192, depth=2, heads=3, dropout=0.1, num_classes=2):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # drop avgpool + fc

        self.project = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.fusion = FeatureFusion(embed_dim)
        self.tokenizer = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # 224x224 input -> ResNet18 conv stack -> 7x7 feature map -> 49 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 49, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.eca = ECABlock(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

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
        x = self.tokenizer(x)

        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = x + self.pos_embed[:, : x.size(1)]

        x = self.transformer(x)
        x = self.eca(x)
        x = self.norm(x)
        x = x.mean(dim=1)

        return self.head(x)
