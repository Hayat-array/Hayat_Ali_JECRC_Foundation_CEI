import os
import io
import base64
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from config import Config
from dataset import load_data, get_train_val_test_splits, add_gaussian_noise
from model import ResidualBlock2D, PatchEmbedding, PositionalEncoding, TransformerEncoderBlock, HybridAutoencoder
from utils import calculate_metrics

app = Flask(__name__, static_folder='static', static_url_path='')

# Configure and setup environment
Config.setup_environment()

# Global variables for dataset and model
X_TEST = None
Y_TEST = None
MODEL = None

def init_dataset():
    global X_TEST, Y_TEST
    try:
        print("[INFO] Loading dataset for API presets...")
        (x_train_full, y_train_full), (x_test_full, y_test_full) = load_data()
        _, _, _, _, X_TEST, Y_TEST = get_train_val_test_splits(
            x_train_full, y_train_full, x_test_full, y_test_full, 
            val_split=Config.VAL_SPLIT, seed=Config.SEED
        )
        print(f"[INFO] Dataset loaded. Total test samples: {X_TEST.shape[0]}")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")

FAST_MODEL_PATH = os.path.join(Config.SAVED_MODEL_DIR, 'fast_denoiser.keras')

def get_model():
    global MODEL
    if MODEL is None:
        # Prefer the optimized fast_denoiser.keras if it exists (better accuracy)
        if os.path.exists(FAST_MODEL_PATH):
            model_path = FAST_MODEL_PATH
            print(f"[INFO] Loading optimized fast denoiser from {model_path}...")
            # compile=False skips custom loss deserialization (inference only)
            MODEL = tf.keras.models.load_model(model_path, compile=False)
        elif os.path.exists(Config.SAVED_MODEL_PATH):
            model_path = Config.SAVED_MODEL_PATH
            print(f"[INFO] Loading hybrid autoencoder from {model_path}...")
            from model import ResidualBlock2D, PatchEmbedding, PositionalEncoding, TransformerEncoderBlock, HybridAutoencoder
            MODEL = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    "ResidualBlock2D": ResidualBlock2D,
                    "PatchEmbedding": PatchEmbedding,
                    "PositionalEncoding": PositionalEncoding,
                    "TransformerEncoderBlock": TransformerEncoderBlock,
                    "HybridAutoencoder": HybridAutoencoder
                }
            )
        else:
            raise FileNotFoundError("No trained model found. Please run fast_train.py first.")
        print(f"[INFO] Model loaded: {type(MODEL).__name__}")
    return MODEL

def run_denoising(model, clean_img: np.ndarray, noise_factor: float) -> tuple:
    """
    Single-pass denoising with unsharp mask sharpening on output.
    Returns (noisy_img, denoised_img) both in shape (1, 28, 28, 1).
    Sharpening formula: output = clip(2*pred - blur(pred), 0, 1)
    which amplifies edges without introducing new artifacts.
    """
    noisy = add_gaussian_noise(clean_img, noise_factor=noise_factor, seed=random.randint(0, 999999))
    pred  = model.predict(noisy, verbose=0)  # shape (1, 28, 28, 1)

    # Unsharp masking in numpy: sharpen = orig + alpha*(orig - blur)
    # Use a simple box-blur approximation via scipy-free convolution
    pred_sq = pred.squeeze()  # (28, 28)
    # Blur with 3x3 average kernel
    blurred = np.zeros_like(pred_sq)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            shifted = np.roll(np.roll(pred_sq, dy, axis=0), dx, axis=1)
            blurred += shifted
    blurred /= 9.0
    # Apply unsharp mask with strength 0.6
    sharpened = pred_sq + 0.6 * (pred_sq - blurred)
    sharpened = np.clip(sharpened, 0.0, 1.0).reshape(1, 28, 28, 1).astype(np.float32)

    return noisy, sharpened

def numpy_to_base64_png(arr: np.ndarray, scale_to: int = None) -> str:
    """Convert a 2D float32 numpy array [0,1] to a sharpened base64-encoded PNG string."""
    from PIL import ImageFilter
    arr = np.clip(arr, 0.0, 1.0)
    img_uint8 = (arr * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')
    if scale_to:
        pil_img = pil_img.resize((scale_to, scale_to), Image.NEAREST)
        # Apply sharpening after upscaling for crisper output
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/sample', methods=['GET'])
def get_sample():
    global X_TEST, Y_TEST
    if X_TEST is None or Y_TEST is None:
        return jsonify({"error": "Dataset not initialized"}), 500

    digit = request.args.get('digit', default=None, type=int)
    
    if digit is not None:
        if digit < 0 or digit > 9:
            return jsonify({"error": "Digit must be between 0 and 9"}), 400
        indices = np.where(Y_TEST == digit)[0]
    else:
        indices = np.arange(len(X_TEST))
        
    if len(indices) == 0:
        return jsonify({"error": f"No samples found for digit {digit}"}), 404
        
    random_idx = random.choice(indices)
    clean_image = X_TEST[random_idx].squeeze()  # Shape: (28, 28)
    label = Y_TEST[random_idx]
    
    return jsonify({
        "image": clean_image.tolist(),
        "label": int(label)
    })

@app.route('/api/denoise', methods=['POST'])
def denoise():
    try:
        data = request.json or {}
        image_data = data.get('image')
        noise_factor = data.get('noise_factor', Config.NOISE_FACTOR)
        
        if image_data is None:
            return jsonify({"error": "Image data is required"}), 400
            
        clean_img = np.array(image_data, dtype=np.float32).reshape(1, 28, 28, 1)
        
        # Add Gaussian noise
        noisy_img = add_gaussian_noise(clean_img, noise_factor=noise_factor, seed=random.randint(0, 100000))
        
        # Load model and predict
        try:
            model = get_model()
        except FileNotFoundError:
            return jsonify({
                "error": "Model file not found. The model is currently training in the background. Please wait a minute and try again."
            }), 503
        
        # Run optimized single-pass denoising with unsharp mask
        noisy_img, reconstructed_img = run_denoising(model, clean_img, noise_factor)
        
        # Calculate metrics
        metrics = calculate_metrics(clean_img, reconstructed_img)
        
        clean_sq = clean_img.squeeze()
        noisy_sq = noisy_img.squeeze()
        recon_sq = reconstructed_img.squeeze()
        
        # Prepare response - both raw 28x28 and full-res PNG
        return jsonify({
            "noisy": noisy_sq.tolist(),
            "reconstructed": recon_sq.tolist(),
            "noisy_png": numpy_to_base64_png(noisy_sq, scale_to=280),
            "reconstructed_png": numpy_to_base64_png(recon_sq, scale_to=280),
            "clean_png": numpy_to_base64_png(clean_sq, scale_to=280),
            "metrics": metrics
        })
        
    except Exception as e:
        print(f"[ERROR] Denoise error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/denoise-image', methods=['POST'])
def denoise_image():
    """Accept a full-resolution uploaded image (base64), preprocess to 28x28,
    run through the denoising model, then return full-resolution results."""
    try:
        data = request.json or {}
        image_b64 = data.get('image_b64')
        noise_factor = data.get('noise_factor', Config.NOISE_FACTOR)
        
        if image_b64 is None:
            return jsonify({"error": "image_b64 field is required"}), 400
        
        # Decode the uploaded image
        image_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('L')  # Grayscale
        original_size = pil_img.size  # (W, H) for resize back later
        
        # Downsample to 28x28
        img_small = pil_img.resize((28, 28), Image.LANCZOS)
        clean_arr = np.array(img_small, dtype=np.float32) / 255.0  # (28, 28)
        clean_img = clean_arr.reshape(1, 28, 28, 1)
        
        # Add Gaussian noise
        noisy_img = add_gaussian_noise(clean_img, noise_factor=noise_factor, seed=random.randint(0, 100000))
        
        # Load model and run inference
        try:
            model = get_model()
        except FileNotFoundError:
            return jsonify({
                "error": "Model file not found. Please wait for the model to finish training."
            }), 503
        
        # Run optimized single-pass denoising with unsharp mask
        noisy_img, reconstructed_img = run_denoising(model, clean_img, noise_factor)
        
        # Calculate metrics at 28x28 level
        metrics = calculate_metrics(clean_img, reconstructed_img)
        
        # Upscale results back to display size (max 560px)
        display_size = min(max(original_size[0], original_size[1]), 560)
        
        clean_sq = clean_img.squeeze()
        noisy_sq = noisy_img.squeeze()
        recon_sq = reconstructed_img.squeeze()
        
        return jsonify({
            "noisy": noisy_sq.tolist(),
            "reconstructed": recon_sq.tolist(),
            "noisy_png": numpy_to_base64_png(noisy_sq, scale_to=display_size),
            "reconstructed_png": numpy_to_base64_png(recon_sq, scale_to=display_size),
            "clean_png": numpy_to_base64_png(clean_sq, scale_to=display_size),
            "metrics": metrics
        })
        
    except Exception as e:
        print(f"[ERROR] Denoise-image error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize dataset at startup
    init_dataset()

    # Pre-load model at startup so first request is fast
    try:
        get_model()
    except FileNotFoundError:
        print("[WARN] No trained model found yet — will load on first request once training completes.")

    # Run server
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
