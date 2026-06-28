import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import kagglehub

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_gaussian_noise(images: np.ndarray, noise_factor: float = 0.25, seed: int = 42) -> np.ndarray:
    """
    Applies additive Gaussian noise to normalized images.
    
    Args:
        images: Clean images normalized to [0.0, 1.0].
        noise_factor: Multiplier for the standard normal distribution.
        seed: Random seed for reproducibility.
        
    Returns:
        Noisy images clipped to [0.0, 1.0].
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = images + noise_factor * noise
    return np.clip(noisy_images, 0.0, 1.0).astype(np.float32)

def detect_and_load_from_dir(directory: str) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
    Scans a directory for CSV or NPZ files containing MNIST data, and parses them.
    
    Args:
        directory: Local directory path to scan.
        
    Returns:
        A tuple of ((x_train, y_train), (x_test, y_test)) or None if parsing fails.
    """
    logger.info(f"Scanning directory for dataset files: {directory}")
    
    # 1. Search for .npz files
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(root, file)
                logger.info(f"Found NPZ file: {file_path}")
                try:
                    with np.load(file_path, allow_pickle=True) as data:
                        # Standard keys for MNIST npz
                        keys = list(data.keys())
                        logger.info(f"NPZ keys: {keys}")
                        
                        x_train_key = next((k for k in keys if 'x_train' in k or 'train_img' in k or 'train_data' in k), None)
                        y_train_key = next((k for k in keys if 'y_train' in k or 'train_lbl' in k or 'train_labels' in k), None)
                        x_test_key = next((k for k in keys if 'x_test' in k or 'test_img' in k or 'test_data' in k), None)
                        y_test_key = next((k for k in keys if 'y_test' in k or 'test_lbl' in k or 'test_labels' in k), None)
                        
                        if x_train_key and x_test_key:
                            x_train = data[x_train_key]
                            x_test = data[x_test_key]
                            y_train = data[y_train_key] if y_train_key else np.zeros(len(x_train))
                            y_test = data[y_test_key] if y_test_key else np.zeros(len(x_test))
                            return (x_train, y_train), (x_test, y_test)
                except Exception as e:
                    logger.error(f"Failed to read NPZ file {file}: {e}")
                    
    # 2. Search for CSV files
    train_csv = None
    test_csv = None
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                if 'train' in file.lower():
                    train_csv = file_path
                elif 'test' in file.lower() or 't10k' in file.lower():
                    test_csv = file_path
                    
    if train_csv:
        try:
            logger.info(f"Found Train CSV: {train_csv}")
            train_df = pd.read_csv(train_csv)
            # Standard MNIST CSV: label, pixel0, pixel1...
            if train_df.shape[1] == 785:
                y_train = train_df.iloc[:, 0].values
                x_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28)
            else:
                x_train = train_df.values.reshape(-1, 28, 28)
                y_train = np.zeros(len(x_train))
                
            if test_csv:
                logger.info(f"Found Test CSV: {test_csv}")
                test_df = pd.read_csv(test_csv)
                if test_df.shape[1] == 785:
                    y_test = test_df.iloc[:, 0].values
                    x_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28)
                else:
                    x_test = test_df.values.reshape(-1, 28, 28)
                    y_test = np.zeros(len(x_test))
            else:
                # If no test set, split train set
                split_idx = int(len(x_train) * 0.8)
                x_test = x_train[split_idx:]
                y_test = y_train[split_idx:]
                x_train = x_train[:split_idx]
                y_train = y_train[:split_idx]
                
            return (x_train, y_train), (x_test, y_test)
        except Exception as e:
            logger.error(f"Failed to read CSV files: {e}")
            
    return None

def load_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Attempts to download MNIST from Kagglehub, falling back to tf.keras.datasets.mnist if necessary.
    
    Returns:
        A tuple of ((x_train, y_train), (x_test, y_test)) containing normalized [0.0, 1.0] float32 image arrays
        of shape (N, 28, 28, 1) and labels.
    """
    try:
        logger.info("Attempting to download awsaf49/mnist-dataset from Kagglehub...")
        path = kagglehub.dataset_download("awsaf49/mnist-dataset")
        logger.info(f"Kagglehub download path: {path}")
        
        kaggle_data = detect_and_load_from_dir(path)
        if kaggle_data is not None:
            logger.info("Successfully loaded and parsed dataset from Kagglehub.")
            (x_train, y_train), (x_test, y_test) = kaggle_data
        else:
            raise ValueError("No compatible dataset files found in Kagglehub path.")
            
    except Exception as e:
        logger.warning(f"Kagglehub download/load failed: {e}. Falling back to tf.keras.datasets.mnist.load_data().")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        logger.info("Successfully loaded MNIST from tf.keras.datasets.")

    # Normalize pixel values to [0.0, 1.0] and convert to float32
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Ensure shape is (N, 28, 28, 1)
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, axis=-1)
    if len(x_test.shape) == 3:
        x_test = np.expand_dims(x_test, axis=-1)
        
    return (x_train, y_train), (x_test, y_test)

def get_train_val_test_splits(
    x_train_full: np.ndarray, 
    y_train_full: np.ndarray, 
    x_test_full: np.ndarray, 
    y_test_full: np.ndarray,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into Train, Validation, and Test sets based on the configuration.
    
    Args:
        x_train_full: All training images.
        y_train_full: All training labels.
        x_test_full: All test images.
        y_test_full: All test labels.
        val_split: Fraction of train data to allocate to validation.
        seed: Random seed for splitting.
        
    Returns:
        Tuple of (x_train, y_train, x_val, y_val, x_test, y_test).
    """
    num_val = int(len(x_train_full) * val_split)
    
    # Shuffle training data
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_train_full))
    
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    
    x_val = x_train_full[val_indices]
    y_val = y_train_full[val_indices]
    
    return x_train, y_train, x_val, y_val, x_test_full, y_test_full

def print_dataset_statistics(name: str, images: np.ndarray):
    """Prints basic statistical details of a given image dataset."""
    print(f"--- Statistics for {name} ---")
    print(f"Shape:      {images.shape}")
    print(f"Data type:  {images.dtype}")
    print(f"Min value:  {images.min():.4f}")
    print(f"Max value:  {images.max():.4f}")
    print(f"Mean value: {images.mean():.4f}")
    print(f"Std value:  {images.std():.4f}")
    print("-" * (len(name) + 22))

def plot_samples(clean: np.ndarray, noisy: np.ndarray, num_samples: int = 5, save_path: str = None):
    """
    Plots a side-by-side comparison of clean and noisy samples.
    
    Args:
        clean: Array of clean images.
        noisy: Array of noisy images.
        num_samples: Number of sample pairs to plot.
        save_path: Optional file path to save the plotted figure.
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        # Clean images
        axes[0, i].imshow(clean[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Clean Images", loc='left', fontsize=10, fontweight='bold')
            
        # Noisy images
        axes[1, i].imshow(noisy[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Noisy Images", loc='left', fontsize=10, fontweight='bold')
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved dataset visualization to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Test script execution
    from config import Config
    Config.setup_environment()
    
    logger.info("Testing Dataset module...")
    (x_train_f, y_train_f), (x_test_f, y_test_f) = load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = get_train_val_test_splits(
        x_train_f, y_train_f, x_test_f, y_test_f, val_split=Config.VAL_SPLIT, seed=Config.SEED
    )
    
    # Generate noise
    x_train_noisy = add_gaussian_noise(x_train, noise_factor=Config.NOISE_FACTOR, seed=Config.SEED)
    
    # Stats
    print_dataset_statistics("Clean Training Data", x_train)
    print_dataset_statistics("Noisy Training Data", x_train_noisy)
    
    # Visualization
    plot_path = os.path.join(Config.IMAGES_DIR, "dataset_samples.png")
    plot_samples(x_train, x_train_noisy, num_samples=5, save_path=plot_path)
