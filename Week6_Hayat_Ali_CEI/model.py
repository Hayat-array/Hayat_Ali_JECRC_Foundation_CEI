import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, Any
from transformer import PatchEmbedding, PositionalEncoding, TransformerEncoderBlock
from config import Config

class ResidualBlock2D(layers.Layer):
    """
    Standard ResNet bottleneck-style Residual Block for 2D convolutions.
    Consists of Conv2D -> Batch Normalization -> ReLU -> Conv2D -> Batch Normalization -> Add -> ReLU.
    """
    def __init__(self, filters: int, kernel_size: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.shortcut = None

    def build(self, input_shape: tf.TensorShape):
        input_channels = input_shape[-1]
        # Match channel dimension if shortcut has a mismatch
        if input_channels != self.filters:
            self.shortcut = layers.Conv2D(self.filters, kernel_size=1, padding='same')
        else:
            self.shortcut = lambda x: x
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        shortcut = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        
        # Add residual connection
        out = layers.add([shortcut, out])
        out = self.relu(out)
        return out

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size
        })
        return config


class HybridAutoencoder(tf.keras.Model):
    """
    Hybrid CNN + Vision Transformer Autoencoder for Image Denoising.
    
    1. CNN Encoder: Extracts hierarchical local features and downsamples the image.
    2. ViT Bottleneck: Learns global attention mappings over the final spatial feature grid.
    3. CNN Decoder: Upsamples features back to original size, leveraging skip connections 
       from the encoder to preserve structural details.
    """
    def __init__(
        self, 
        input_shape_config: tuple = Config.INPUT_SHAPE,
        d_model: int = Config.D_MODEL,
        num_heads: int = Config.NUM_HEADS,
        num_layers: int = Config.NUM_LAYERS,
        mlp_dim: int = Config.MLP_DIM,
        dropout_rate: float = Config.DROPOUT,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_shape_config = input_shape_config
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        # --- CNN Encoder ---
        # Stage 1: 28x28x1 -> 28x28x32
        self.enc_conv1 = layers.Conv2D(32, kernel_size=3, padding='same')
        self.enc_bn1 = layers.BatchNormalization()
        self.enc_relu1 = layers.ReLU()
        self.enc_res1 = ResidualBlock2D(32)
        
        # Stage 2: 28x28x32 -> 14x14x64
        self.enc_conv2 = layers.Conv2D(64, kernel_size=3, padding='same')
        self.enc_bn2 = layers.BatchNormalization()
        self.enc_relu2 = layers.ReLU()
        self.enc_pool2 = layers.MaxPooling2D(pool_size=2)
        self.enc_res2 = ResidualBlock2D(64)
        
        # Stage 3: 14x14x64 -> 7x7x128
        self.enc_conv3 = layers.Conv2D(128, kernel_size=3, padding='same')
        self.enc_bn3 = layers.BatchNormalization()
        self.enc_relu3 = layers.ReLU()
        self.enc_pool3 = layers.MaxPooling2D(pool_size=2)
        self.enc_res3 = ResidualBlock2D(128)
        
        self.enc_dropout = layers.Dropout(dropout_rate)

        # --- Transformer Bottleneck ---
        self.patch_embed = PatchEmbedding(d_model=d_model)
        # 7x7 spatial resolution = 49 tokens
        self.pos_enc = PositionalEncoding(num_patches=49, d_model=d_model)
        
        self.transformer_blocks = [
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout_rate,
                name=f"vit_block_{i}"
            ) for i in range(num_layers)
        ]
        
        self.dec_projection = layers.Dense(128) # Project to match decoder features

        # --- CNN Decoder ---
        # Stage 1 Upsampling (7x7 -> 14x14)
        self.dec_up1 = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')
        self.dec_bn_up1 = layers.BatchNormalization()
        self.dec_relu_up1 = layers.ReLU()
        self.dec_conv_reduce1 = layers.Conv2D(64, kernel_size=1, padding='same') # Adjust concatenated channels
        self.dec_res1 = ResidualBlock2D(64)
        
        # Stage 2 Upsampling (14x14 -> 28x28)
        self.dec_up2 = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')
        self.dec_bn_up2 = layers.BatchNormalization()
        self.dec_relu_up2 = layers.ReLU()
        self.dec_conv_reduce2 = layers.Conv2D(32, kernel_size=1, padding='same') # Adjust concatenated channels
        self.dec_res2 = ResidualBlock2D(32)
        
        # Output layers
        self.dec_final_conv = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # --- Encoder ---
        x1 = self.enc_conv1(inputs)
        x1 = self.enc_bn1(x1, training=training)
        x1 = self.enc_relu1(x1)
        x1 = self.enc_res1(x1, training=training)      # Save for decoder skip 2
        
        x2 = self.enc_conv2(x1)
        x2 = self.enc_bn2(x2, training=training)
        x2 = self.enc_relu2(x2)
        x2 = self.enc_pool2(x2)
        x2 = self.enc_res2(x2, training=training)      # Save for decoder skip 1
        
        x3 = self.enc_conv3(x2)
        x3 = self.enc_bn3(x3, training=training)
        x3 = self.enc_relu3(x3)
        x3 = self.enc_pool3(x3)
        x3 = self.enc_res3(x3, training=training)      # Output of CNN Encoder (7x7x128)
        
        x_enc_out = self.enc_dropout(x3, training=training)

        # --- Transformer Bottleneck ---
        x_tokens = self.patch_embed(x_enc_out)
        x_tokens = self.pos_enc(x_tokens)
        
        for block in self.transformer_blocks:
            x_tokens = block(x_tokens, training=training)
            
        x_bottleneck = self.dec_projection(x_tokens)
        
        # Reshape back to 2D: (batch, 7, 7, 128)
        batch_size = tf.shape(inputs)[0]
        x_bottleneck_2d = tf.reshape(x_bottleneck, (batch_size, 7, 7, 128))

        # --- Decoder ---
        # Upsample bottleneck: 7x7x128 -> 14x14x64
        u1 = self.dec_up1(x_bottleneck_2d, training=training)
        u1 = self.dec_bn_up1(u1, training=training)
        u1 = self.dec_relu_up1(u1)
        
        # Skip connection 1: concatenate with x2 (14x14x64)
        c1 = layers.concatenate([u1, x2], axis=-1)     # Shape: 14x14x128
        c1 = self.dec_conv_reduce1(c1)                  # Shape: 14x14x64
        d1 = self.dec_res1(c1, training=training)       # Shape: 14x14x64
        
        # Upsample: 14x14x64 -> 28x28x32
        u2 = self.dec_up2(d1, training=training)
        u2 = self.dec_bn_up2(u2, training=training)
        u2 = self.dec_relu_up2(u2)
        
        # Skip connection 2: concatenate with x1 (28x28x32)
        c2 = layers.concatenate([u2, x1], axis=-1)     # Shape: 28x28x64
        c2 = self.dec_conv_reduce2(c2)                  # Shape: 28x28x32
        d2 = self.dec_res2(c2, training=training)       # Shape: 28x28x32
        
        # Final reconstruction (sigmoid to map to [0, 1])
        outputs = self.dec_final_conv(d2)              # Shape: 28x28x1
        outputs = tf.cast(outputs, tf.float32)          # Cast for mixed precision stability
        return outputs

    def get_config(self) -> Dict[str, Any]:
        return {
            "input_shape_config": self.input_shape_config,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate
        }


def build_model(config: Config = Config) -> tf.keras.Model:
    """Instantiates and compiles the hybrid autoencoder."""
    model = HybridAutoencoder(
        input_shape_config=config.INPUT_SHAPE,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        mlp_dim=config.MLP_DIM,
        dropout_rate=config.DROPOUT
    )
    
    # Run a forward pass with dry-run data to define input/output shapes in the model
    dummy_input = tf.zeros((1, *config.INPUT_SHAPE))
    _ = model(dummy_input)
    
    # We compile the model using AdamW optimizer and Mean Squared Error Loss
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mae']
    )
    
    return model


if __name__ == "__main__":
    print("[INFO] Verifying Model module...")
    Config.setup_environment()
    
    # Build and summarize model
    model = build_model(Config)
    model.summary()
    
    # Test random forward pass
    test_batch = tf.random.normal((4, 28, 28, 1))
    test_output = model(test_batch)
    print("Input Batch Shape: ", test_batch.shape)
    print("Output Batch Shape:", test_output.shape)
    assert test_output.shape == (4, 28, 28, 1)
    print("[INFO] Model validation successful!")
