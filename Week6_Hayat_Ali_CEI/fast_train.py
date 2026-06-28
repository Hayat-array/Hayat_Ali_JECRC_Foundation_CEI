"""
fast_train.py — Optimized CPU training for high-quality denoising
=================================================================
Strategy:
  - 10,000 training samples (fast enough on CPU)
  - Combined MSE + SSIM loss for sharper, structurally correct outputs
  - Dynamic noise augmentation: random noise factor per batch (0.1 - 0.5)
    so the model generalizes across noise levels
  - Higher learning rate + cosine decay
  - 15 epochs with aggressive early stopping
Target: PSNR > 19 dB, SSIM > 0.85 on the test set
"""

import os
import random
import logging
import numpy as np
import tensorflow as tf
from config import Config
from dataset import load_data, get_train_val_test_splits

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

os.makedirs(Config.SAVED_MODEL_DIR, exist_ok=True)
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# ── Combined Loss: MSE + SSIM ──────────────────────────────────────────────────
@tf.keras.saving.register_keras_serializable(package='FastDenoiser')
def ssim_loss(y_true, y_pred):
    """1 - SSIM: lower is better, maximises structural similarity."""
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

@tf.keras.saving.register_keras_serializable(package='FastDenoiser')
def combined_loss(y_true, y_pred):
    """80% MSE + 20% SSIM loss for balanced pixel accuracy and structure."""
    mse  = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = ssim_loss(y_true, y_pred)
    return 0.80 * mse + 0.20 * ssim

# ── Noise Augmentation Generator ──────────────────────────────────────────────
def make_aug_dataset(x_clean, batch_size=128, min_noise=0.05, max_noise=0.50):
    """
    Returns a tf.data.Dataset that applies a RANDOM noise factor
    per batch so the model becomes noise-level agnostic.
    """
    ds = tf.data.Dataset.from_tensor_slices(x_clean)
    ds = ds.shuffle(buffer_size=len(x_clean), seed=SEED)
    ds = ds.batch(batch_size)

    def add_random_noise(clean_batch):
        # Sample a single noise factor for the whole batch
        nf = tf.random.uniform((), minval=min_noise, maxval=max_noise)
        noise = tf.random.normal(tf.shape(clean_batch), stddev=nf)
        noisy = tf.clip_by_value(clean_batch + noise, 0.0, 1.0)
        return noisy, clean_batch

    ds = ds.map(add_random_noise, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ── Build Efficient Model ──────────────────────────────────────────────────────
def build_fast_model():
    """
    Lightweight but effective CNN denoising autoencoder.
    Uses depthwise separable convolutions for speed + skip connections for quality.
    No Transformer bottleneck (too slow on CPU for large datasets).
    Target: PSNR > 19 dB at noise=0.25
    """
    inp = tf.keras.Input(shape=(28, 28, 1))

    # ─ Encoder ─
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    skip1 = x                                                   # 28×28×32

    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    skip2 = x                                                   # 14×14×64

    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)                 # 7×7×128

    # ─ Bottleneck (dense attention-lite) ─
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # ─ Decoder ─
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, skip2])              # 14×14×128
    x = tf.keras.layers.Conv2D(64, 1, padding='same', activation='relu')(x)

    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([x, skip1])              # 28×28×64
    x = tf.keras.layers.Conv2D(32, 1, padding='same', activation='relu')(x)

    # Final output: sigmoid maps to [0,1]
    out = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    model = tf.keras.Model(inp, out, name='FastDenoiser')

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=2e-3,
            decay_steps=15 * 80,   # 15 epochs × ~80 steps (10K / 128)
            alpha=1e-5
        )
    )
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=['mae'])
    return model


def main():
    logger.info("=== Fast CPU Training for High-Accuracy Denoising ===")

    # ── Load data ────────────────────────────────────────────────────
    logger.info("Loading MNIST dataset...")
    (x_train_full, y_train_full), (x_test_full, y_test_full) = load_data()

    _, _, _, _, x_test, _ = get_train_val_test_splits(
        x_train_full, y_train_full, x_test_full, y_test_full,
        val_split=0.1, seed=SEED
    )

    # Use 10,000 training samples and 1,000 validation samples for speed
    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(len(x_train_full), size=10000, replace=False)
    val_idx   = rng.choice(len(x_test_full),  size=1000,  replace=False)

    x_train = x_train_full[train_idx]
    x_val   = x_test_full[val_idx]

    logger.info(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

    # ── Build augmenting datasets ─────────────────────────────────────
    train_ds = make_aug_dataset(x_train, batch_size=128)
    # Fixed val noise at 0.25 for consistent monitoring
    from dataset import add_gaussian_noise
    x_val_noisy = add_gaussian_noise(x_val, noise_factor=0.25, seed=99)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val_noisy, x_val)).batch(128).prefetch(tf.data.AUTOTUNE)

    # ── Build model ───────────────────────────────────────────────────
    logger.info("Building fast CNN denoising model...")
    model = build_fast_model()
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────
    SAVE_PATH = os.path.join(Config.SAVED_MODEL_DIR, 'fast_denoiser.keras')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=SAVE_PATH,
            monitor='val_loss', save_best_only=True,
            save_weights_only=False, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=4,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.4, patience=2,
            min_lr=1e-6, verbose=1
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("Starting training (15 epochs max)...")
    history = model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # ── Evaluate on full test set at noise=0.25 ───────────────────────
    logger.info("Evaluating on full test set...")
    x_test_noisy = add_gaussian_noise(x_test, noise_factor=0.25, seed=200)
    x_test_pred  = model.predict(x_test_noisy, batch_size=256, verbose=1)

    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    psnr_list, ssim_list, mse_list = [], [], []
    for i in range(len(x_test)):
        c = x_test[i]
        p = np.clip(x_test_pred[i], 0, 1)
        mse_list.append(float(np.mean((c - p)**2)))
        psnr_list.append(float(peak_signal_noise_ratio(c, p, data_range=1.0)))
        ssim_list.append(float(structural_similarity(c.squeeze(), p.squeeze(), data_range=1.0)))

    print("\n" + "="*50)
    print("   FAST DENOISER — TEST SET RESULTS")
    print("="*50)
    print(f"  MSE  : {np.mean(mse_list):.5f}")
    print(f"  PSNR : {np.mean(psnr_list):.2f} dB")
    print(f"  SSIM : {np.mean(ssim_list):.4f}")
    print("="*50)
    print(f"\nModel saved to: {SAVE_PATH}")


if __name__ == '__main__':
    main()
