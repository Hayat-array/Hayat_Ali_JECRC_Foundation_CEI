import os
import argparse
import logging
import tensorflow as tf
from config import Config
from dataset import load_data, get_train_val_test_splits, add_gaussian_noise
from model import build_model
from utils import plot_loss_curves

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    # 1. Override Config settings with arguments if provided
    if args.epochs:
        Config.EPOCHS = args.epochs
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.noise_factor is not None:
        Config.NOISE_FACTOR = args.noise_factor
    if args.learning_rate:
        Config.LEARNING_RATE = args.learning_rate

    # 2. Setup Environment (directories & seeds)
    Config.setup_environment()
    
    # 3. Enable Mixed Precision training if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled: mixed_float16")
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")
            
    # 4. Load Data
    logger.info("Loading MNIST dataset...")
    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_data()
    
    # Split into Train, Validation, and Test
    x_train, y_train, x_val, y_val, x_test, y_test = get_train_val_test_splits(
        x_train_full, y_train_full, x_test_full, y_test_full, 
        val_split=Config.VAL_SPLIT, seed=Config.SEED
    )
    
    logger.info(f"Splits summary - Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")
    
    # Subset data for quick-train if requested
    if getattr(args, 'quick_train', False):
        logger.info("Quick-train mode enabled. Subsetting dataset to 1000 train, 200 validation, and 200 test samples...")
        x_train = x_train[:1000]
        y_train = y_train[:1000]
        x_val = x_val[:200]
        y_val = y_val[:200]
        x_test = x_test[:200]
        y_test = y_test[:200]
        logger.info(f"Quick-splits summary - Train: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")
    
    # 5. Generate Noisy Datasets
    logger.info(f"Generating Gaussian noise with noise factor = {Config.NOISE_FACTOR}...")
    x_train_noisy = add_gaussian_noise(x_train, noise_factor=Config.NOISE_FACTOR, seed=Config.SEED)
    x_val_noisy = add_gaussian_noise(x_val, noise_factor=Config.NOISE_FACTOR, seed=Config.SEED + 1)
    
    # 6. Create tf.data.Dataset Pipelines
    train_ds = tf.data.Dataset.from_tensor_slices((x_train_noisy, x_train))
    train_ds = train_ds.shuffle(buffer_size=10000).batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((x_val_noisy, x_val))
    val_ds = val_ds.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 7. Instantiate and Compile Model
    logger.info("Building Hybrid CNN-ViT Autoencoder model...")
    model = build_model(Config)
    model.summary()
    
    # 8. Setup Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=Config.SAVED_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False, # Save complete model architecture & weights
        verbose=1
    )
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # 9. Model Training
    logger.info(f"Starting training for {Config.EPOCHS} epochs with batch size {Config.BATCH_SIZE}...")
    history = model.fit(
        train_ds,
        epochs=Config.EPOCHS,
        validation_data=val_ds,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler_cb]
    )
    
    # 10. Post-Training Visualizations and Plots
    loss_curve_path = os.path.join(Config.OUTPUT_DIR, "loss_curve.png")
    logger.info(f"Plotting training loss curves to {loss_curve_path}...")
    plot_loss_curves(history.history, loss_curve_path)
    
    logger.info("Training pipeline execution successfully completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid CNN + Vision Transformer Autoencoder for Image Denoising")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--noise-factor", type=float, default=None, help="Gaussian noise multiplier")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate for AdamW")
    parser.add_argument("--quick-train", action="store_true", help="Subset data for quick execution on CPU")
    
    args = parser.parse_args()
    main(args)
