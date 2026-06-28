import os
import random
import numpy as np
import tensorflow as tf

class Config:
    # Reproducibility
    SEED = 42

    # Dataset Settings
    INPUT_SHAPE = (28, 28, 1)
    NOISE_FACTOR = 0.25
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

    # Training Settings
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Transformer Bottleneck Configuration
    # Input to bottleneck will be (batch, 7, 7, 128)
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    MLP_DIM = 256
    DROPOUT = 0.1

    # Directories and Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
    SAVED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "autoencoder.keras")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    IMAGES_DIR = os.path.join(BASE_DIR, "images")

    @classmethod
    def setup_environment(cls):
        """Configure directory structures and set seeds for reproducibility."""
        # Create directories if they do not exist
        os.makedirs(cls.SAVED_MODEL_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.IMAGES_DIR, exist_ok=True)

        # Set random seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        tf.random.set_seed(cls.SEED)
        tf.experimental.numpy.random.seed(cls.SEED)
        
        # Verify GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[INFO] GPUs detected: {len(gpus)}. Using GPU acceleration.")
            # Enable memory growth to avoid allocating all GPU memory at once
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"[WARNING] Error configuring GPU memory growth: {e}")
        else:
            print("[INFO] No GPUs detected. Running on CPU.")
