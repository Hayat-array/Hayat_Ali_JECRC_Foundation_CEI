import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict, Any

class PatchEmbedding(layers.Layer):
    """
    Custom layer that projects spatial feature maps into a 1D sequence of token embeddings.
    For an input feature map of shape (batch_size, H, W, C), this layer flattens the spatial 
    dimensions to (H * W) and projects the channels to d_model.
    """
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.projection = layers.Dense(d_model)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Get dynamic shapes
        shape = tf.shape(x)
        batch_size = shape[0]
        h, w, c = shape[1], shape[2], shape[3]
        
        # Reshape to (batch_size, H * W, C)
        x_flat = tf.reshape(x, (batch_size, h * w, c))
        
        # Project to (batch_size, H * W, d_model)
        return self.projection(x_flat)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


class PositionalEncoding(layers.Layer):
    """
    Custom layer that adds learnable positional encodings to a token sequence.
    """
    def __init__(self, num_patches: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.d_model = d_model
        self.pos_emb = None

    def build(self, input_shape: tf.TensorShape):
        self.pos_emb = self.add_weight(
            name="pos_emb",
            shape=(1, self.num_patches, self.d_model),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x is expected to have shape (batch_size, num_patches, d_model)
        return x + self.pos_emb

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "d_model": self.d_model
        })
        return config


class TransformerEncoderBlock(layers.Layer):
    """
    A single Vision Transformer Encoder Block using pre-layer normalization (Pre-LN).
    """
    def __init__(self, d_model: int, num_heads: int, mlp_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout

        # Pre-LN Normalization layers
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

        # Multi-Head Self-Attention Layer
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads, 
            dropout=dropout
        )

        # Feed Forward Network (FFN)
        self.ffn = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout)
        ])

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # First Pre-LN Attention Block with Residual Connection
        x_norm = self.ln1(x, training=training)
        attn_out = self.mha(query=x_norm, value=x_norm, key=x_norm, training=training)
        x = x + attn_out

        # Second Pre-LN MLP Block with Residual Connection
        x_norm2 = self.ln2(x, training=training)
        ffn_out = self.ffn(x_norm2, training=training)
        x = x + ffn_out

        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout_rate
        })
        return config


if __name__ == "__main__":
    # Test script execution
    print("[INFO] Verifying Transformer module layers...")
    
    # Create test input of shape (batch, H, W, C)
    test_input = tf.random.normal((8, 7, 7, 128))
    print("Input Shape:", test_input.shape)
    
    # 1. Test Patch Embedding
    patch_embed_layer = PatchEmbedding(d_model=128)
    x_emb = patch_embed_layer(test_input)
    print("After PatchEmbedding:", x_emb.shape)
    assert x_emb.shape == (8, 49, 128)
    
    # 2. Test Positional Encoding
    pos_enc_layer = PositionalEncoding(num_patches=49, d_model=128)
    x_pos = pos_enc_layer(x_emb)
    print("After PositionalEncoding:", x_pos.shape)
    assert x_pos.shape == (8, 49, 128)
    
    # 3. Test Transformer Block
    tx_block = TransformerEncoderBlock(d_model=128, num_heads=4, mlp_dim=256, dropout=0.1)
    x_tx = tx_block(x_pos)
    print("After TransformerEncoderBlock:", x_tx.shape)
    assert x_tx.shape == (8, 49, 128)
    
    print("[INFO] Custom Transformer layers successfully verified!")
