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
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.perceiver = None
        self.fallback_encoder = None
        self.input_projection = None
        
        if PerceiverModel is not None:
            try:
                # Load with low_cpu_mem_usage to reduce memory spikes
                self.perceiver = PerceiverModel.from_pretrained(model_name)
                # Get actual latent dim from model config if available
                if hasattr(self.perceiver.config, 'd_latents'):
                    self.latent_dim = self.perceiver.config.d_latents
                    print(f"Perceiver latent dimension: {self.latent_dim}")
                else:
                    raise ValueError("Could not find Perceiver latent dimension in config")

                if hasattr(self.perceiver.config, 'num_latents'):
                    self.num_latents = self.perceiver.config.num_latents
                    print(f"Perceiver number of latents: {self.num_latents}")
                else:
                    raise ValueError("Could not find Perceiver number of latents in config")
                
                # Get Perceiver's expected input dimension
                if hasattr(self.perceiver.config, 'd_model'):
                    perceiver_input_dim = self.perceiver.config.d_model
                elif hasattr(self.perceiver.config, 'd_input'):
                    perceiver_input_dim = self.perceiver.config.d_input
                else:
                    # Default fallback
                    perceiver_input_dim = 704
                    print(f"Warning: Could not find Perceiver input dimension in config, using default: {perceiver_input_dim}")
                
                # Create projection layer if input_dim is provided and different from Perceiver's expected dim
                if input_dim is not None and input_dim != perceiver_input_dim:
                    self.input_projection = nn.Linear(input_dim, perceiver_input_dim)
                    print(f"Created input projection: {input_dim} -> {perceiver_input_dim}")
                else:
                    print(f"Perceiver input dimension: {perceiver_input_dim}, LLM embedding dimension: {input_dim}")
            except Exception as e:
                print(f"Warning: Could not load Perceiver IO model '{model_name}'. Error: {e}")
                print("Creating a simple MLP-based encoder as fallback.")
                raise e

        self.compressor = CrossAttentionCompressor(
            latent_dim=self.latent_dim,
            num_heads=8,
        )
    
    def freeze_base_model(self):
        """
        Freeze all parameters in the base PerceiverModel.
        The input_projection, CrossAttentionCompressor and projection layer added later remain trainable.
        """
        if self.perceiver is None:
            return
        
        # Freeze all parameters in the base PerceiverModel
        for param in self.perceiver.parameters():
            param.requires_grad = False
        
        # Keep input_projection trainable (it's a new layer we added)
        if self.input_projection is not None:
            for param in self.input_projection.parameters():
                param.requires_grad = True
        
        # Print summary
        total = sum(p.numel() for p in self.perceiver.parameters())
        print(f"Frozen base PerceiverModel: {total:,} parameters")
        if self.input_projection is not None:
            input_proj_params = sum(p.numel() for p in self.input_projection.parameters())
            print(f"Input projection layer (trainable): {input_proj_params:,} parameters")
        
    def forward(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Process inputs through Perceiver IO.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Optional attention mask
            
        Returns:
            Latent representations of shape (batch_size, num_latents, latent_dim)
        """
        # Project inputs to Perceiver's expected dimension if needed
        if self.input_projection is not None:
            inputs = self.input_projection(inputs)
        
        try:
            # Perceiver IO API may vary - adjust based on actual implementation
            outputs = self.perceiver(
                inputs=inputs,
                attention_mask=attention_mask
            )
            outputs = outputs.last_hidden_state
            return self.compressor(outputs)
        except Exception as e:
            print(f"Warning: Perceiver IO forward pass failed: {e}")
            raise e
