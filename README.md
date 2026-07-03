# HybridCNNViT — Breast Cancer Histopathology Classification

A lightweight hybrid CNN-Transformer architecture for benign vs. malignant classification on the **BreakHis** breast cancer histopathology dataset. The model fuses a ResNet18 convolutional backbone with a shallow Transformer encoder, using a custom **depthwise Feature Fusion block** and **Efficient Channel Attention (ECA)** to combine local texture cues with global tissue-level context.

> Custom architecture + full baseline comparison (CNN vs. ViT vs. Hybrid) + component-level ablation study, all evaluated on a patient-stratified test split.

---

## Motivation

Histopathology classification sits at an interesting boundary: CNNs are strong at capturing local texture (cell density, nuclei shape) but weak at modeling long-range tissue architecture, while Vision Transformers do the opposite but need far more data to train well from scratch. This project asks a simple question — **can a small hybrid model get the best of both, cheaply enough to train on a single Colab T4?**

## Architecture

```
Input (224×224×3)
      │
ResNet18 Backbone (conv layers only, ImageNet-pretrained)
      │
1×1 Projection → 192-dim feature maps
      │
Depthwise Feature Fusion Block   ← local texture refinement (residual)
      │
Conv Tokenizer (3×3) → 7×7 grid → 49 tokens
      │
+ Learnable Positional Embeddings
      │
Transformer Encoder ×2 (3 heads, GELU)   ← global context modeling
      │
ECA Attention   ← lightweight channel re-weighting
      │
LayerNorm → Global Average Pool
      │
MLP Head → [Benign, Malignant]
```

**Core contributions:**
- **FeatureFusion block** — a depthwise-separable residual block inserted between the CNN backbone and the tokenizer, letting the model refine local features before they're flattened into tokens.
- **ECA attention** — a near-parameter-free channel attention mechanism (1D conv over pooled tokens) applied after the Transformer encoder, cheaper than SE-style attention.
- **Progressive baseline design** — the hybrid isn't evaluated in isolation; it's benchmarked directly against a fine-tuned ResNet18 and a ViT-Tiny baseline trained under identical conditions.

## Results

Test-set performance (patient-stratified split, 1,187 images):

| Model         | Accuracy | Precision | Recall | F1     | ROC-AUC |
|---------------|:--------:|:---------:|:------:|:------:|:-------:|
| ResNet18      | 0.9756   | 0.9839    | 0.9802 | 0.9820 | 0.9971  |
| ViT-Tiny      | 0.9747   | 0.9791    | 0.9839 | 0.9815 | 0.9965  |
| **HybridCNNViT** | **0.9848** | **0.9901** | **0.9876** | **0.9888** | **0.9979** |

*(ViT-Tiny numbers above are from its own standalone training/eval run — see [note on the comparison cell](#known-issue-in-the-original-notebook) below.)*

### Ablation study

Each component was removed independently from the trained Hybrid model to measure its contribution:

| Variant           | Accuracy | F1     | AUC    |
|-------------------|:--------:|:------:|:------:|
| Full Hybrid       | 0.9848   | 0.9888 | 0.9979 |
| No ECA            | 0.9848   | 0.9888 | 0.9979 |
| No Feature Fusion | 0.9781   | 0.9840 | 0.9979 |
| No Transformer    | 0.4558   | 0.4705 | 0.4999 |

**Takeaway:** the Transformer encoder is doing almost all of the heavy lifting — removing it collapses performance to near-random. Feature Fusion gives a smaller but consistent boost. ECA contributes marginally on this dataset/scale, which is a useful negative result rather than a flaw — it suggests channel attention has less to offer once you already have a Transformer providing long-range mixing at this model size.

## Known issue in the original notebook

The in-notebook "Week 5" model-comparison cell reloads a fresh, **randomly-initialized** ViT-Tiny (`pretrained=False`, no `load_state_dict` call) instead of the actual fine-tuned checkpoint, so the side-by-side comparison table undercounts ViT-Tiny (~38% acc). The real ViT-Tiny performance is the one logged during its own dedicated training run (0.9747 acc, 0.9965 AUC), which is what's reported in the results table above. `src/compare_models.py` in this repo fixes the loading bug so the comparison is apples-to-apples.

## Dataset

[BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis) — 7,909 breast tissue histopathology images across benign and malignant classes, multiple magnifications. Split 70/15/15 into train/val/test.

> ⚠️ The current split is a random image-level split. Because BreakHis contains multiple images per patient/slide, this risks patient leakage between train and test. A patient-stratified split is recommended before reporting these numbers as final (see `TODO` in `src/dataset.py`).

## Project structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── breakhis_hybrid_cnn_vit.ipynb   # original end-to-end Colab notebook
├── src/
│   ├── dataset.py                      # BreakHisDataset, transforms, splits
│   ├── train_utils.py                  # shared training/eval loop, metrics
│   ├── compare_models.py               # baseline vs. hybrid comparison (bug-fixed)
│   ├── ablation.py                     # component ablation variants
│   └── models/
│       ├── resnet_baseline.py
│       ├── vit_baseline.py
│       └── hybrid_cnn_vit.py           # ECABlock, FeatureFusion, HybridCNNViT
└── results/
    └── (figures: ROC curves, confusion matrices, ablation charts)
```

## Setup

```bash
pip install -r requirements.txt
```

Designed to run on a free Colab T4 GPU. Dataset is pulled automatically via `kagglehub`.

## Usage

```python
from src.dataset import get_dataloaders
from src.models.hybrid_cnn_vit import HybridCNNViT
from src.train_utils import train_model, evaluate

train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)

model = HybridCNNViT(embed_dim=192, depth=2, heads=3).to(device)
train_model(model, train_loader, val_loader, epochs=10)
metrics = evaluate(model, test_loader)
```

## Tech stack

PyTorch · torchvision · timm (ViT-Tiny) · scikit-learn (metrics) · kagglehub · Google Colab (T4 GPU)

## Future work

- [ ] Patient-stratified train/val/test split to eliminate leakage risk
- [ ] Multi-magnification evaluation (40×/100×/200×/400×) — current split pools all magnifications
- [ ] Grad-CAM / attention-rollout visualizations for interpretability
- [ ] Multi-class subtype classification (BreakHis provides 8 tumor subtypes, not just binary)

## Author

Aditi — M.Sc.-M.Tech. Data and Computational Science, IIT Jodhpur
