import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logger = logging.getLogger(__name__)

def calculate_metrics(clean: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Computes Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), 
    and Structural Similarity Index (SSIM) between clean and reconstructed images.
    
    Metrics are computed per image in the batch and averaged.
    """
    # Ensure inputs are float32 in range [0, 1]
    clean = np.clip(clean, 0.0, 1.0).astype(np.float32)
    reconstructed = np.clip(reconstructed, 0.0, 1.0).astype(np.float32)
    
    num_images = clean.shape[0]
    mse_list = []
    psnr_list = []
    ssim_list = []
    
    for i in range(num_images):
        c_img = clean[i]
        r_img = reconstructed[i]
        
        # Calculate MSE
        mse_val = np.mean((c_img - r_img) ** 2)
        mse_list.append(mse_val)
        
        # Calculate PSNR
        try:
            psnr_val = peak_signal_noise_ratio(c_img, r_img, data_range=1.0)
            psnr_list.append(psnr_val)
        except Exception as e:
            # Fallback to analytical calculation if skimage fails
            if mse_val > 0:
                psnr_val = 20 * np.log10(1.0 / np.sqrt(mse_val))
            else:
                psnr_val = 100.0
            psnr_list.append(psnr_val)
            
        # Calculate SSIM
        try:
            # Try newer skimage channel_axis API
            ssim_val = structural_similarity(c_img, r_img, data_range=1.0, channel_axis=-1)
        except TypeError:
            try:
                # Try older multichannel parameter
                ssim_val = structural_similarity(c_img, r_img, data_range=1.0, multichannel=True)
            except TypeError:
                # Squeeze channel dimension if 2D
                ssim_val = structural_similarity(c_img.squeeze(), r_img.squeeze(), data_range=1.0)
        ssim_list.append(ssim_val)
        
    return {
        "mse": float(np.mean(mse_list)),
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list))
    }

def plot_loss_curves(history: Dict[str, list], save_path: str):
    """
    Plots the training and validation loss curves from training history.
    """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(history['loss']) + 1)
    
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
    plt.title('Training and Validation Loss Curves', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss (MSE)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Loss curve saved successfully at {save_path}")
    plt.close()

def plot_denoising_results(
    clean: np.ndarray, 
    noisy: np.ndarray, 
    reconstructed: np.ndarray, 
    save_path: str, 
    num_samples: int = 5
):
    """
    Plots original, noisy, reconstructed images, and difference heatmaps side-by-side.
    """
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 2.2, 8.5))
    
    # Clip and convert values
    clean = np.clip(clean, 0.0, 1.0)
    noisy = np.clip(noisy, 0.0, 1.0)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)
    diff = np.abs(clean - reconstructed)
    
    for i in range(num_samples):
        # 1. Clean image
        axes[0, i].imshow(clean[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original", loc='center', fontsize=11, fontweight='bold')
            
        # 2. Noisy image
        axes[1, i].imshow(noisy[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Noisy Input", loc='center', fontsize=11, fontweight='bold')
            
        # 3. Reconstructed image
        axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title("Reconstructed", loc='center', fontsize=11, fontweight='bold')
            
        # 4. Difference Heatmap
        im = axes[3, i].imshow(diff[i].squeeze(), cmap='inferno')
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_title("Diff Heatmap", loc='center', fontsize=11, fontweight='bold')
            
    # Add a global colorbar for the difference maps
    fig.subplots_adjust(right=0.88, wspace=0.1, hspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.08, 0.02, 0.18])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_title("Error", fontsize=9, fontweight='bold')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison plot saved successfully at {save_path}")
    plt.close()

def save_metrics_text(metrics: Dict[str, float], save_path: str):
    """
    Saves a dictionary of metrics formatted nicely in a text file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("=========================================\n")
        f.write("  Denoising Performance Evaluation Metrics\n")
        f.write("=========================================\n")
        for key, val in metrics.items():
            f.write(f"{key.upper():<10} : {val:.6f}\n")
        f.write("=========================================\n")
    logger.info(f"Evaluation metrics saved in text format at {save_path}")
