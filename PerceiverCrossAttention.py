"""
Cross attention block for attending to Perceiver IO outputs.
This module performs cross-attention where queries come from the Qwen model
and keys/values come from the Perceiver model outputs.

Two implementations:
1. PerceiverCrossAttention: Custom implementation with GQA support
2. PerceiverCrossAttentionSimple: Simplified version using nn.MultiheadAttention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PerceiverCrossAttention(nn.Module):
    """
    Simplified cross attention block using nn.MultiheadAttention.
    Use this if you don't need Grouped Query Attention (GQA).
    """
    def __init__(
        self,
        hidden_size: int,
        perceiver_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_size: Hidden dimension of the Qwen model
            perceiver_dim: Dimension of Perceiver outputs (latent_dim)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.perceiver_dim = perceiver_dim
        
        # Project Perceiver outputs to hidden_size if needed
        if perceiver_dim != hidden_size:
            self.perceiver_proj = nn.Linear(perceiver_dim, hidden_size, bias=False)
        else:
            self.perceiver_proj = nn.Identity()
        
        # Use built-in MultiheadAttention for cross-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Learned gating to blend cross-attention output with residual
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        perceiver_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of cross-attention.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - queries from Qwen layer
            perceiver_outputs: (batch_size, num_latents, perceiver_dim) - Perceiver outputs (keys/values)
            attention_mask: Optional attention mask (will be converted to key_padding_mask format)
        
        Returns:
            output: (batch_size, seq_len, hidden_size) - cross-attention output
            attention_weights: Optional attention weights for debugging
        """
        # Project Perceiver outputs to hidden_size
        perceiver_proj = self.perceiver_proj(perceiver_outputs)
        
        # nn.MultiheadAttention expects key_padding_mask where True means ignore
        # For cross-attention, the mask applies to keys (Perceiver outputs)
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask shape: (batch_size, seq_len, num_latents)
            # For key_padding_mask, we need (batch_size, num_latents) where True = ignore
            # Take the mask for the first sequence position (or average if needed)
            if attention_mask.dim() == 3:
                # Collapse seq_len dimension - use first position's mask
                key_padding_mask = (attention_mask[:, 0, :] == 0)
            elif attention_mask.dim() == 2:
                # Already in (batch, num_latents) format
                key_padding_mask = (attention_mask == 0)
        
        # Cross-attention: query from hidden_states, key/value from perceiver outputs
        attn_output, attn_weights = self.attention(
            query=hidden_states,
            key=perceiver_proj,
            value=perceiver_proj,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        
        # Apply learned gating (sigmoid gate for stability)
        gate_weight = torch.sigmoid(self.gate)
        attn_output = gate_weight * attn_output
        
        return attn_output, attn_weights

