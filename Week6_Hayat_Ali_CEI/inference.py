import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from config import Config
from dataset import load_data, get_train_val_test_splits, add_gaussian_noise
from model import ResidualBlock2D, PatchEmbedding, PositionalEncoding, TransformerEncoderBlock, HybridAutoencoder
from utils import calculate_metrics, plot_denoising_results, save_metrics_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    # 1. Setup Environment
    Config.setup_environment()
    
    # Check model existence
    model_path = args.model_path if args.model_path else Config.SAVED_MODEL_PATH
    if not os.path.exists(model_path):
        logger.error(f"Saved model file not found at {model_path}. Please train the model first using train.py.")
        return
        
    # 2. Load Dataset
    logger.info("Loading dataset for evaluation...")
    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_data()
    
    # Split to get the consistent test set
    _, _, _, _, x_test, y_test = get_train_val_test_splits(
        x_train_full, y_train_full, x_test_full, y_test_full, 
        val_split=Config.VAL_SPLIT, seed=Config.SEED
    )
    
    # Generate noisy test set
    noise_factor = args.noise_factor if args.noise_factor is not None else Config.NOISE_FACTOR
    logger.info(f"Generating noisy test set with noise factor = {noise_factor}...")
    x_test_noisy = add_gaussian_noise(x_test, noise_factor=noise_factor, seed=Config.SEED + 2)
    
    # 3. Load Trained Model with Custom Objects
    logger.info(f"Loading trained autoencoder model from {model_path}...")
    custom_objects = {
        "ResidualBlock2D": ResidualBlock2D,
        "PatchEmbedding": PatchEmbedding,
        "PositionalEncoding": PositionalEncoding,
        "TransformerEncoderBlock": TransformerEncoderBlock,
        "HybridAutoencoder": HybridAutoencoder
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 4. Perform Inference / Prediction
    logger.info("Running denoising inference on test dataset...")
    x_test_reconstructed = model.predict(x_test_noisy, batch_size=Config.BATCH_SIZE, verbose=1)
    
    # 5. Evaluate Metrics
    logger.info("Calculating performance metrics (MSE, PSNR, SSIM)...")
    metrics = calculate_metrics(x_test, x_test_reconstructed)
    
    print("\n" + "=" * 40)
    print("       TEST SET EVALUATION RESULTS")
    print("=" * 40)
    print(f"MSE  : {metrics['mse']:.6f}")
    print(f"PSNR : {metrics['psnr']:.4f} dB")
    print(f"SSIM : {metrics['ssim']:.4f}")
    print("=" * 40 + "\n")
    
    # 6. Save Reports and Comparisons
    # Save text report
    report_path = os.path.join(Config.OUTPUT_DIR, "psnr_ssim.txt")
    save_metrics_text(metrics, report_path)
    
    # Save visual grid comparison
    comparison_path = os.path.join(Config.OUTPUT_DIR, "comparison.png")
    logger.info(f"Generating visual denoising comparison grid at {comparison_path}...")
    
    # Pick random samples for plotting
    rng = np.random.default_rng(Config.SEED)
    sample_indices = rng.choice(len(x_test), size=5, replace=False)
    
    plot_denoising_results(
        clean=x_test[sample_indices],
        noisy=x_test_noisy[sample_indices],
        reconstructed=x_test_reconstructed[sample_indices],
        save_path=comparison_path,
        num_samples=5
    )
    
    # Save separate copies in images/ directory for documentation
    doc_comparison_path = os.path.join(Config.IMAGES_DIR, "denoising_results.png")
    plot_denoising_results(
        clean=x_test[sample_indices],
        noisy=x_test_noisy[sample_indices],
        reconstructed=x_test_reconstructed[sample_indices],
        save_path=doc_comparison_path,
        num_samples=5
    )
    
    logger.info("Inference and evaluation script successfully executed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Hybrid CNN + ViT Autoencoder for Image Denoising")
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model file (.keras)")
    parser.add_argument("--noise-factor", type=float, default=None, help="Noise factor override for evaluation")
    
    args = parser.parse_args()
    main(args)
