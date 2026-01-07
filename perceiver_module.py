"""
Perceiver IO module for processing conversation turns into latent representations.
"""
import torch
import torch.nn as nn
from typing import Optional

from transformers import PerceiverModel


class CrossAttentionCompressor(nn.Module):
    """
    Cross-attention module that compresses multiple latent vectors into a single vector.
    Uses learnable queries to attend over the latent vectors.
    """
    def __init__(
        self,
        latent_dim: int,
        num_queries: int = 1,
        num_heads: int = 8,
    ):
        """
        Args:
            latent_dim: Dimension of input latent vectors
            output_dim: Dimension of output vector (defaults to latent_dim)
            num_queries: Number of learnable query vectors (default 1 for single output)
            num_heads: Number of attention heads
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_queries = num_queries
        
        # Learnable query vectors
        self.query = nn.Parameter(torch.randn(1, num_queries, latent_dim))
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Compress latent vectors into a single vector using cross-attention.
        
        Args:
            latents: Input latent vectors of shape (batch_size, num_latents, latent_dim)
            
        Returns:
            Compressed vector of shape (batch_size, num_queries, output_dim)
        """
        batch_size = latents.shape[0]
        queries = self.query.expand(batch_size, -1, -1)  # (batch_size, num_queries, latent_dim)
        attended, _ = self.cross_attn(queries, latents, latents)  # (batch_size, num_queries, latent_dim)
        
        return attended

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
        use_compressor: bool = True,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.perceiver = None
        self.input_projection = None
        self.compressor = None

        self.perceiver = PerceiverModel.from_pretrained(model_name)
        self.latent_dim = self.perceiver.config.d_latents
        print(f"Perceiver latent dimension: {self.latent_dim}")
        self.num_latents = self.perceiver.config.num_latents
        print(f"Perceiver number of latents: {self.num_latents}")
        
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
        
        # Add cross-attention compressor to convert latents to single vector
        if use_compressor:
            self.compressor = CrossAttentionCompressor(
                latent_dim=self.latent_dim,
            )
            print(f"Created CrossAttentionCompressor: {self.num_latents} latents -> {1} vector(s)")

        if output_dim is not None:
            self.output_projection = nn.Linear(self.latent_dim, output_dim)
            print(f"Created output projection: {self.latent_dim} -> {output_dim}")
    
    def freeze_base_model(self):
        """
        Freeze all parameters in the base PerceiverModel.
        The input_projection and compressor layers remain trainable.
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
        
        # Keep compressor trainable (it's a new layer we added)
        if self.compressor is not None:
            for param in self.compressor.parameters():
                param.requires_grad = True
        
        # Print summary
        total = sum(p.numel() for p in self.perceiver.parameters())
        print(f"Frozen base PerceiverModel: {total:,} parameters")
        if self.input_projection is not None:
            input_proj_params = sum(p.numel() for p in self.input_projection.parameters())
            print(f"Input projection layer (trainable): {input_proj_params:,} parameters")
        if self.compressor is not None:
            compressor_params = sum(p.numel() for p in self.compressor.parameters())
            print(f"CrossAttentionCompressor (trainable): {compressor_params:,} parameters")
        if self.output_projection is not None:
            output_proj_params = sum(p.numel() for p in self.output_projection.parameters())
            print(f"Output projection layer (trainable): {output_proj_params:,} parameters")

    def forward(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Process inputs through Perceiver IO and optionally compress to single vector.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Optional attention mask
            
        Returns:
            If use_compressor=True: Compressed vector(s) of shape (batch_size, num_queries, output_dim)
            If use_compressor=False: Latent representations of shape (batch_size, num_latents, latent_dim)
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
            latents = outputs.last_hidden_state  # (batch_size, num_latents, latent_dim)
            
            # If compressor is enabled, compress latents to single vector
            if self.compressor is not None:
                latents = self.compressor(latents)  # (batch_size, num_queries, output_dim)
            
            if self.output_projection is not None:
                latents = self.output_projection(latents)  # (batch_size, num_queries, output_dim)
            
            return latents
            
        except Exception as e:
            print(f"Warning: Perceiver IO forward pass failed: {e}")
            raise e
