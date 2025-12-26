import torch
import torch.nn as nn
from typing import Optional, Tuple
import torch.nn.functional as F

class LatentCrossAttention(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        input_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        num_latents: int = 784,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_latents = num_latents
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        
        # GQA ratio: how many query heads share each KV head
        self.num_groups = num_query_heads // num_kv_heads
        assert num_query_heads % num_kv_heads == 0, \
            f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    
        
        # Query projection from model hidden states
        self.q_proj = nn.Linear(hidden_size, num_query_heads * self.head_dim, bias=False)
        
        # Key-Value projections from memory embeddings
        # Memory can have different dimension, so we project to model hidden_size first
        self.memory_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(num_query_heads * self.head_dim, hidden_size, bias=False)
        
        # Learned gating to blend memory output with block output
        self.gate = nn.Parameter(torch.zeros(1))

        # Learnable scale to control memory strength (initialized > 1)
        # This makes the memory path noticeably affect activations even before training.
        self.mem_scale = nn.Parameter(torch.ones(1) * 15.0)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) - queries from transformer block
            memory_embeddings: (batch_size, memory_size, hidden_size) - external memory vectors
            attention_mask: Optional (batch_size, seq_len, memory_size) attention mask
        
        Returns:
            output: (batch_size, seq_len, hidden_size) - cross-attention output
            attention_weights: (batch_size, num_query_heads, seq_len, memory_size) - attention weights
        """
        batch_size, seq_len, _ = hidden_states.shape
        _, memory_len, _ = memory_embeddings.shape
        
        # Project queries from hidden states
        # (batch_size, seq_len, num_query_heads * head_dim)
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch_size, num_query_heads, seq_len, head_dim)
        
        # Apply positional encoding to memory embeddings
        # Expand positional encoding to batch: (memory_size, hidden_size) -> (batch_size, memory_size, hidden_size)
        pos_encoding = self.memory_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        # Only use the first memory_len positions (in case memory_len < memory_size)
        pos_encoding = pos_encoding[:, :memory_len, :]
        # Add positional encoding to memory embeddings
        memory_embeddings = memory_embeddings + pos_encoding
        
        # Project memory embeddings to model dimension, then to K, V
        # (batch_size, memory_len, hidden_size)
        memory_proj = self.memory_proj(memory_embeddings)
        
        # (batch_size, memory_len, num_kv_heads * head_dim)
        k = self.k_proj(memory_proj)
        k = k.view(batch_size, memory_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_kv_heads, memory_len, head_dim)
        
        v = self.v_proj(memory_proj)
        v = v.view(batch_size, memory_len, self.num_kv_heads, self.head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_kv_heads, memory_len, head_dim)
        
        # Expand KV heads to match Q heads for GQA
        # Repeat each KV head num_groups times
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)  # (batch_size, num_query_heads, memory_len, head_dim)
            v = v.repeat_interleave(self.num_groups, dim=1)  # (batch_size, num_query_heads, memory_len, head_dim)
        
        # Compute attention scores: Q @ K^T / sqrt(d)
        # (batch_size, num_query_heads, seq_len, memory_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (batch_size, seq_len, memory_len) -> (batch_size, 1, seq_len, memory_len)
            attn_scores = attn_scores.masked_fill(
                attention_mask.unsqueeze(1) == 0, float('-inf')
            )
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch_size, num_query_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_query_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.num_query_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)  # (batch_size, seq_len, hidden_size)
        
        # Apply learned gating (sigmoid gate for stability) and scaling
        gate_weight = torch.sigmoid(self.gate)
        attn_output = gate_weight * self.mem_scale * attn_output

        # mem_norm = attn_output.norm(dim=-1).mean().item()
        # hid_norm = hidden_states.norm(dim=-1).mean().item()
        # print(f"[mem debug] mem_norm={mem_norm:.4f}, hidden_norm={hid_norm:.4f}")
        
        return attn_output, attn_weights