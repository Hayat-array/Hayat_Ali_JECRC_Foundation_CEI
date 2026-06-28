# Hybrid CNN + Vision Transformer Autoencoder for Image Denoising

This repository contains a production-quality, end-to-end implementation of a **Hybrid CNN + Vision Transformer (ViT) Autoencoder** designed to denoise images. Specifically, it removes additive Gaussian noise from handwritten digits using the MNIST dataset. 

The architecture is implemented from scratch in Python using TensorFlow 2.x and Keras, without relying on external transformer libraries.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architectural Details](#architectural-details)
3. [Mathematical Intuition](#mathematical-intuition)
4. [Project Structure](#project-structure)
5. [Installation & Requirements](#installation--requirements)
6. [Workflow Guide](#workflow-guide)
   - [Centralized Configuration](#centralized-configuration)
   - [Training Module](#training-module)
   - [Inference & Evaluation](#inference--evaluation)
   - [Jupyter Notebook](#jupyter-notebook)
7. [Expected Results](#expected-results)
8. [Future Improvements](#future-improvements)

---

## Project Overview

Image denoising is a fundamental computer vision task that aims to reconstruct a clean image $\mathbf{X}$ from its noise-corrupted observation $\mathbf{Y} = \mathbf{X} + \mathbf{N}$.

While traditional Convolutional Autoencoders (CAEs) excel at capturing local spatial details (edges, textures) due to translation equivariance and local receptive fields, they often fail to capture long-range dependencies and global contexts. On the other hand, pure Vision Transformers (ViTs) model global contexts using self-attention, but they are computationally expensive and lack inductive biases for local patterns, requiring massive datasets to generalize.

This project implements a **Hybrid CNN + Vision Transformer Autoencoder**:
- **CNN Encoder**: Downsamples the input image and extracts rich localized spatial features.
- **Transformer Bottleneck**: Flattens the low-resolution feature map into sequence tokens, applying multi-head self-attention to capture global contexts and global structural coherence.
- **CNN Decoder**: Reconstructs the high-resolution clean image using skip connections from the encoder stages to preserve high-frequency details.

```
       [Input Noisy Image] (28x28x1)
               │
         [CNN Encoder] (Extracts local features & downsamples)
               │   └─ Skip Connection 1 (28x28x32) ───┐
               │   └─ Skip Connection 2 (14x14x64) ───┼─┐
               ▼                                       │ │
      [Bottleneck Feature Map] (7x7x128)               │ │
               │                                       │ │
     [Transformer Bottleneck] (Learns global attention) │ │
               │                                       │ │
      [Reshaped Bottleneck] (7x7x128)                  │ │
               │                                       │ │
         [CNN Decoder] (Upsamples & restores detail) ◄─┴─┘
               │
      [Denoised Output] (28x28x1)
```

---

## Architectural Details

### 1. CNN Encoder
The encoder downsamples the $(28, 28, 1)$ inputs into a $(7, 7, 128)$ feature map through three stages:
- **Stage 1**: $3\times3$ Conv2D (32 filters) $\rightarrow$ Batch Normalization $\rightarrow$ ReLU $\rightarrow$ Residual Block (32 filters). 
- **Stage 2**: $3\times3$ Conv2D (64 filters, stride 2 or MaxPooling) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ Residual Block (64 filters). Shape: $(14, 14, 64)$.
- **Stage 3**: $3\times3$ Conv2D (128 filters, stride 2 or MaxPooling) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ Residual Block (128 filters) $\rightarrow$ Dropout. Shape: $(7, 7, 128)$.

*Note: The intermediate representations from Stage 1 and Stage 2 are saved as skip connections to feed into corresponding stages of the decoder.*

### 2. Transformer Bottleneck
The spatial feature map of shape $(H_f, W_f, C_f) = (7, 7, 128)$ is processed as a sequence:
- **Patch Embedding**: The $7 \times 7$ grid is flattened spatially to $49$ tokens. A linear projection (Dense layer) projects them to `d_model = 128`.
- **Positional Encoding**: Learnable 1D positional encodings of shape $(1, 49, 128)$ are added to the sequence to preserve spatial relationship information.
- **Transformer Encoder Block**: Built using $2$ layers of custom Transformer blocks. Each block employs:
  - Pre-Layer Normalization (`LayerNormalization`).
  - Multi-Head Self Attention (`MultiHeadAttention`) with $4$ heads and key dimension of $32$ per head.
  - Feed Forward Network (`FFN`) consisting of two Dense layers (sizes $256$ and $128$) with GELU activation.
  - Residual connections around both the attention and FFN layers.

### 3. CNN Decoder
The sequence output of the Transformer is projected back to $(7 \times 7, 128)$, reshaped to $(7, 7, 128)$, and decoded:
- **Upsample Stage 1**: Conv2DTranspose upsamples $(7, 7, 128)$ to $(14, 14, 64)$. This is concatenated with the Stage 2 encoder skip connection (shape $14\times14\times64$) resulting in $128$ channels. A $1\times1$ Conv reduces channels back to $64$, followed by a Residual Block.
- **Upsample Stage 2**: Conv2DTranspose upsamples $(14, 14, 64)$ to $(28, 28, 32)$. This is concatenated with the Stage 1 encoder skip connection (shape $28\times28\times32$) resulting in $64$ channels. A $1\times1$ Conv reduces channels back to $32$, followed by a Residual Block.
- **Output Stage**: A final $3\times3$ Conv2D with a Sigmoid activation maps the channels back to $(28, 28, 1)$, matching the original image dimensions.

---

## Mathematical Intuition

### 1. Additive Gaussian Noise Model
The corrupted input image is modeled as:
$$\mathbf{Y} = \mathbf{X} + \eta \cdot \mathbf{Z}$$

where:
- $\mathbf{X} \in [0, 1]^{H \times W \times C}$ represents the clean ground truth.
- $\mathbf{Z} \sim \mathcal{N}(0, \mathbf{I})$ is standard white Gaussian noise.
- $\eta$ is the noise factor (standard deviation of the noise), configured as `0.25`.

### 2. Multi-Head Self Attention
Self-attention maps a query ($Q$) and a set of keys ($K$) and values ($V$) to an output:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

For Multi-Head Attention, the queries, keys, and values are linearly projected $h$ times with different learnable projections:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

where $d_k = d_{model} / h$ is the dimensionality of key projections.

### 3. Reconstruction Loss Function
The model parameters $\theta$ are optimized by minimizing the Mean Squared Error (MSE) between clean images $X_i$ and reconstructed outputs $\hat{X}_i = f_\theta(Y_i)$:
$$\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{N \cdot H \cdot W \cdot C} \sum_{i=1}^N \sum_{j=1}^{H \cdot W \cdot C} \left( X_{i, j} - \hat{X}_{i, j} \right)^2$$

### 4. Evaluation Metrics
We use two standardized computer vision metrics alongside MSE to evaluate denoising quality:
- **Peak Signal-to-Noise Ratio (PSNR)**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
  $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right) = 20 \cdot \log_{10} \left( \frac{1.0}{\sqrt{\text{MSE}}} \right)$$
- **Structural Similarity Index (SSIM)**: Compares luminance ($l$), contrast ($c$), and structure ($s$) between clean ($x$) and reconstructed ($y$) windows:
  $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

---

## Project Structure

```
Autoencoder-Transformer-Denoising/
├── config.py                 # Central configuration for seeds and hyperparameters
├── dataset.py                # Pipeline for MNIST loading, noise injection, and splits
├── transformer.py            # From-scratch implementation of Custom ViT Layers
├── model.py                  # Full Encoder-Transformer-Decoder Autoencoder network
├── utils.py                  # Evaluation metrics (PSNR, SSIM, MSE) and plotting utilities
├── train.py                  # Script to run mixed-precision model training
├── inference.py              # Script to perform model validation and load weights
├── autoencoder.ipynb         # Interactive Jupyter Notebook displaying end-to-end execution
└── requirements.txt          # Python package requirements
```

---

## Installation & Requirements

### Prerequisites
- Python 3.11 or higher
- CUDA-compatible GPU (Highly recommended for speed, though CPU training is supported)

### Installation Steps
1. **Clone or navigate** to the project workspace:
   ```bash
   cd Autoencoder-Transformer-Denoising
   ```
2. **Install dependencies** using pip:
   ```bash
   pip install -r requirements.txt
   ```

---

## Workflow Guide

### Centralized Configuration
Open `config.py` to customize parameters. The defaults are:
- `NOISE_FACTOR = 0.25`
- `BATCH_SIZE = 64`
- `EPOCHS = 30`
- `D_MODEL = 128` (Transformer embedding size)

### Training Module
To train the model from scratch:
```bash
python train.py --epochs 30 --batch-size 64 --noise-factor 0.25
```
This script will:
- Download the dataset (Kaggle or fallback to Keras).
- Prepare train and validation tf.data.Dataset pipelines.
- Train the model utilizing Mixed Precision (if GPU detected).
- Save checkpoints dynamically to `saved_models/autoencoder.keras`.
- Plot validation curves and save them to `outputs/loss_curve.png`.

### Inference & Evaluation
To run evaluation using the saved weights:
```bash
python inference.py --model-path saved_models/autoencoder.keras
```
This script will:
- Load the model specifying custom layers.
- Evaluate the model on the test dataset.
- Print average MSE, PSNR, and SSIM.
- Save numerical outputs in `outputs/psnr_ssim.txt`.
- Save side-by-side comparison grids in `outputs/comparison.png` and `images/denoising_results.png`.

### Jupyter Notebook
Launch the notebook to inspect the steps interactively:
```bash
jupyter notebook autoencoder.ipynb
```
The notebook walks through the code modules visually, letting you run block-by-block.

---

## Expected Results

When training with a noise factor of `0.25` for 20–30 epochs, the model is expected to converge to high-performance metrics:
- **Test MSE**: $\le 0.012$
- **Test PSNR**: $\ge 19.5 \text{ dB}$ (Typical baseline autoencoders get ~16-17 dB)
- **Test SSIM**: $\ge 0.88$ (Indicating strong structural similarity)

---

## Future Improvements
1. **Adversarial Training**: Introduce a PatchGAN Discriminator to convert the autoencoder into a Denoising GAN (Pix2Pix) for sharper reconstructions.
2. **Channel Attention**: Integrate Squeeze-and-Excitation (SE) blocks within the CNN residual modules.
3. **Multi-scale Skip Connections**: Implement features similar to U-Net++ to bridge encoder/decoder semantic gaps.
