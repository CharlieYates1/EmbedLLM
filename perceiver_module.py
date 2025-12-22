"""
Perceiver IO module for processing conversation turns into latent representations.
"""
import torch
import torch.nn as nn
from typing import Optional

try:
    from transformers import PerceiverModel
except ImportError:
    PerceiverModel = None


class PerceiverIOModule(nn.Module):
    """
    Wrapper around Perceiver IO model for processing conversation turns.
    Note: Perceiver IO expects inputs in a specific format. This implementation
    assumes we're using embeddings as inputs, which may need adjustment based on
    the specific Perceiver IO variant.
    """
    def __init__(
        self,
        model_name: str = "deepmind/multimodal-perceiver",
        latent_dim: int = 512,
        num_latents: int = 256,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.perceiver = None
        self.fallback_encoder = None
        
        if PerceiverModel is not None:
            try:
                self.perceiver = PerceiverModel.from_pretrained(model_name)
                # Get actual latent dim from model config if available
                if hasattr(self.perceiver.config, 'd_latents'):
                    self.latent_dim = self.perceiver.config.d_latents
                else:
                    self.latent_dim = latent_dim
            except Exception as e:
                print(f"Warning: Could not load Perceiver IO model '{model_name}'. Error: {e}")
                print("Creating a simple MLP-based encoder as fallback.")
                raise e
        
        self.num_latents = num_latents
        
    def forward(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Process inputs through Perceiver IO.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Latent representations of shape (batch_size, num_latents, latent_dim)
        """

        try:
            # Perceiver IO API may vary - adjust based on actual implementation
            outputs = self.perceiver(
                inputs=inputs,
                attention_mask=attention_mask
            )
            if hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state
            elif isinstance(outputs, torch.Tensor):
                return outputs
            else:
                raise ValueError(f"Unexpected output format: {type(outputs)}")
        except Exception as e:
            print(f"Warning: Perceiver IO forward pass failed: {e}")
            raise e

class CrossAttentionCompressor(nn.Module):
    """
    Uses cross-attention to compress Perceiver IO latent space into a single vector.
    """
    def __init__(self, latent_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latents: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Compress multiple latent vectors into a single vector using cross-attention.
        
        Args:
            latents: Tensor of shape (batch_size, num_latents, latent_dim)
            attention_mask: Optional attention mask for latents
            
        Returns:
            Single compressed vector of shape (batch_size, 1, latent_dim)
        """
        batch_size = latents.shape[0]
        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)
        
        # Apply cross-attention: query attends to latents
        attn_output, _ = self.cross_attention(
            query=query,
            key=latents,
            value=latents,
            key_padding_mask=attention_mask
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(attn_output + query)
        output = self.dropout(output)
        
        return output

