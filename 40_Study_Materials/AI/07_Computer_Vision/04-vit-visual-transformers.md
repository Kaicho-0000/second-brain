---
tags:
  - computer-vision
  - ViT
  - transformer
  - DeiT
  - Swin-Transformer
created: "2026-04-19"
status: draft
---

# 04 — ViT と視覚 Transformer

## 1. Vision Transformer（ViT）の革新

2020年にDosovitskiy et al. が発表した ViT は、画像を **パッチ列** として扱い、NLP の Transformer をそのまま適用した画像認識モデル。CNN を一切使わずに SOTA 精度を達成した。

```mermaid
flowchart LR
    A["入力画像\n(224×224×3)"] --> B["パッチ分割\n(16×16 パッチ\n= 196個)"]
    B --> C["線形射影\n各パッチ→D次元"]
    C --> D["[CLS]トークン\n+ 位置埋め込み"]
    D --> E["Transformer\nEncoder\n×L層"]
    E --> F["[CLS] の\n出力ベクトル"]
    F --> G["分類ヘッド\n(MLP)"]
    G --> H["クラス予測"]
```

### 1.1 パッチ埋め込み

画像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ を $N = \frac{HW}{P^2}$ 個のパッチに分割:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}}; \mathbf{E}\mathbf{x}_1; \mathbf{E}\mathbf{x}_2; \ldots; \mathbf{E}\mathbf{x}_N] + \mathbf{E}_{\text{pos}}$$

- $\mathbf{E} \in \mathbb{R}^{D \times (P^2 C)}$: 線形射影行列
- $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$: 位置埋め込み

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # 畳み込みでパッチ分割と線形射影を同時実行
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)              # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)    # (B, N+1, D)
        x = x + self.pos_embed
        return x
```

### 1.2 ViT のスケール

| モデル | 層数 | 隠れ次元 | ヘッド数 | パラメータ |
|--------|------|----------|----------|-----------|
| ViT-S/16 | 12 | 384 | 6 | 22M |
| ViT-B/16 | 12 | 768 | 12 | 86M |
| ViT-L/16 | 24 | 1024 | 16 | 307M |
| ViT-H/14 | 32 | 1280 | 16 | 632M |
| ViT-G/14 | 48 | 1664 | 16 | 1.8B |

---

## 2. DeiT（Data-efficient Image Transformer）

### 2.1 課題と解決

ViT は大規模データ（ImageNet-21K, JFT-300M）が必要だったが、DeiT は **ImageNet-1K のみ** で高精度を達成。

**主要な改良点**:
- **蒸留トークン**: [CLS] とは別に [DIST] トークンを追加し、教師モデル（RegNet等）から知識蒸留
- **強力なデータ拡張**: RandAugment, Mixup, CutMix
- **正則化**: Stochastic Depth, Label Smoothing

```python
import timm

# DeiT の利用
model = timm.create_model("deit3_base_patch16_224.fb_in1k", pretrained=True)
```

### 2.2 Hard Distillation vs Soft Distillation

$$\mathcal{L}_{\text{hard}} = \frac{1}{2}\mathcal{L}_{\text{CE}}(y, \psi(Z_s)) + \frac{1}{2}\mathcal{L}_{\text{CE}}(y_t, \psi(Z_s^{\text{dist}}))$$

$$\mathcal{L}_{\text{soft}} = \frac{1}{2}\mathcal{L}_{\text{CE}}(y, \psi(Z_s)) + \frac{1}{2}\tau^2 \text{KL}(\psi(Z_t/\tau), \psi(Z_s^{\text{dist}}/\tau))$$

---

## 3. Swin Transformer

### 3.1 階層的な窓 Attention

ViT の Self-Attention は計算量が $O(N^2)$ で画像サイズに対して2乗スケール。Swin は **局所的な窓（Window）** 内でのみ Attention を計算し、窓をシフトして窓間の情報伝達を実現。

```mermaid
flowchart TD
    subgraph "Swin Transformer の階層構造"
        S1["Stage 1\nパッチ: 56×56\nチャネル: 96"] --> M1["Patch Merging\n2×2統合"]
        M1 --> S2["Stage 2\nパッチ: 28×28\nチャネル: 192"]
        S2 --> M2["Patch Merging"]
        M2 --> S3["Stage 3\nパッチ: 14×14\nチャネル: 384"]
        S3 --> M3["Patch Merging"]
        M3 --> S4["Stage 4\nパッチ: 7×7\nチャネル: 768"]
    end
```

### 3.2 Shifted Window Attention

```python
class WindowAttention(nn.Module):
    """窓ベースの Multi-Head Self-Attention"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # 相対位置バイアス
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # 相対位置バイアスを追加
        attn = attn + self.relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)
```

### 3.3 計算量の比較

| モデル | 計算量（画像サイズ $H$） |
|--------|------------------------|
| ViT | $O(H^4)$ (全ピクセル間 Attention) |
| Swin | $O(H^2 \cdot M^2)$ ($M$: 窓サイズ、通常7) |

---

## 4. その他の視覚 Transformer

### 4.1 BEiT（BERT-style Pre-training for Images）

画像パッチをマスクし、離散化されたビジュアルトークンを予測（Masked Image Modeling）。

### 4.2 MAE（Masked Autoencoder）

パッチの75%をマスクし、ピクセル値を再構成。非常に効率的な事前学習。

### 4.3 FlexiViT

パッチサイズを可変にし、単一モデルで様々な解像度に対応。

---

## 5. CNN vs Transformer の比較

| 特性 | CNN | Vision Transformer |
|------|-----|-------------------|
| 帰納バイアス | 局所性、平行移動不変性 | 少ない（位置埋め込みで対応） |
| データ効率 | 少量データでも動作 | 大規模データが必要 |
| スケーラビリティ | 飽和しやすい | スケールに対して良い |
| 解像度変更 | 柔軟 | 位置埋め込みの補間が必要 |
| 解釈性 | フィルタ可視化 | Attention マップ |

---

## 6. ハンズオン演習

### 演習 1: ViT のパッチ埋め込み可視化

ViT のパッチ埋め込みと位置埋め込みの類似度をヒートマップとして可視化し、位置情報の学習具合を確認せよ。

### 演習 2: Attention マップの可視化

ViT の各層の Attention マップを抽出し、モデルが画像のどこに注目しているかを可視化せよ。[CLS] トークンの Attention パターンの層ごとの変化を分析。

### 演習 3: CNN vs ViT の比較実験

同じデータセット（CIFAR-100）で ResNet-50 と ViT-S/16 をスクラッチ学習し、学習曲線・最終精度・推論速度を比較せよ。

---

## 7. まとめ

- ViT は画像をパッチ列として Transformer に入力する革新的アーキテクチャ
- パッチ埋め込み + 位置埋め込み + [CLS] トークンが基本構造
- DeiT は蒸留と強力な正則化で小規模データでも ViT を学習可能に
- Swin は階層構造と窓 Attention で効率的かつ高精度
- CNN の帰納バイアスは少量データで有利、Transformer は大規模でスケール

---

## 参考文献

- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, 2021)
- Touvron et al., "Training data-efficient image transformers" (DeiT, 2021)
- Liu et al., "Swin Transformer: Hierarchical Vision Transformer" (2021)
