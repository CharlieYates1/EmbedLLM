"""
Integration code for adding memory cross-attention to Qwen3-4B model.
This module provides utilities to modify Qwen3 models from Hugging Face.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from transformers import AutoModelForCausalLM
from perceiver_module import PerceiverIOModule
from PerceiverCrossAttention import PerceiverCrossAttention

class QwenWithPerceiverCrossAttn(nn.Module):
    """
    Wrapper for Qwen3 model with memory cross-attention at layer 7.
    """
    
    def __init__(
        self,
        qwen_model_name: str = "Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged",
        perceiver_model_name: str = "deepmind/multimodal-perceiver",
        layer_index: int = 7,
    ):
        """
        Args:
            model: Qwen3 model from transformers
            layer_index: Layer index (0-based) to insert cross-attention
            memory_size: Number of memory vectors
            insert_after_ffn: If True, insert after feed-forward; if False, after self-attention
            memory_manager: Optional pre-initialized memory manager
        """
        super().__init__()
        
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        self.model_dtype = next(self.qwen_model.parameters()).dtype

        self.config = self.qwen_model.config
        self.hidden_size = self.config.hidden_size
        self.num_query_heads = self.config.num_attention_heads

        self.perceiver = PerceiverIOModule(
            model_name=perceiver_model_name,
            input_dim=self.qwen_model.config.hidden_size,
        )
        self._perceiver_outputs = None
        self.layer_index = layer_index
        
        # Modify the specified layer
        self._freeze_qwen_model()
        self._modify_layer()

    def _freeze_qwen_model(self):
        """Freeze the Qwen model."""
        for param in self.qwen_model.parameters():
            param.requires_grad = False
    
    def _modify_layer(self):
        """Insert cross-attention into the specified layer."""
        # Access transformer layers
        if hasattr(self.qwen_model, 'model') and hasattr(self.qwen_model.model, 'layers'):
            layers = self.qwen_model.model.layers
        elif hasattr(self.qwen_model, 'layers'):
            layers = self.qwen_model.layers
        else:
            raise ValueError("Could not find transformer layers in model")
        
        # Verify layer index
        if self.layer_index < 0 or self.layer_index >= len(layers):
            raise ValueError(
                f"Layer index {self.layer_index} out of range [0, {len(layers)})"
            )
        
        # Get the target layer
        target_layer = layers[self.layer_index]
        
        # Get the device and dtype of the target layer
        layer_device = next(target_layer.parameters()).device
        
        # Create cross-attention module
        cross_attn = PerceiverCrossAttention(
            hidden_size=self.hidden_size,
            perceiver_dim=self.perceiver.latent_dim,
            num_heads=self.num_query_heads,
        )
        
        # Move cross-attention modules to the same device and dtype as the layer
        cross_attn = cross_attn.to(device=layer_device, dtype=self.model_dtype)
        cross_attn_layer_norm = nn.LayerNorm(self.hidden_size).to(device=layer_device, dtype=self.model_dtype)
        
        # Store cross-attention components and reference to memory manager
        target_layer.perceiver_cross_attn = cross_attn
        target_layer.cross_attn_layer_norm = cross_attn_layer_norm
        
        # Patch the forward method
        original_forward = target_layer.forward
        
        def patched_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):  
            # Call original forward (full block)
            outputs = original_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            
            # Extract output (first element is usually hidden_states)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                other_outputs = outputs[1:]
            else:
                hidden_states = outputs
                other_outputs = ()
            
            # Apply cross-attention
            if self._perceiver_outputs is not None:
                cross_attn_output, cross_attn_weights = target_layer.perceiver_cross_attn(
                    hidden_states=hidden_states,
                    perceiver_outputs=self._perceiver_outputs,
                )
                
                # Add residual and layer norm
                hidden_states = target_layer.cross_attn_layer_norm(
                    hidden_states + cross_attn_output
                )
            
            # Return in same format as original
            if isinstance(outputs, tuple):
                return (hidden_states,) + other_outputs
            return hidden_states
        
        target_layer.forward = patched_forward
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass with memory augmentation.
        """
        embed_layer = self.qwen_model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)
        if inputs_embeds.dtype != self.model_dtype:
            inputs_embeds = inputs_embeds.to(dtype=self.model_dtype)
        self._perceiver_outputs = self.perceiver(inputs_embeds)

        # Standard forward through model
        outputs = self.qwen_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        
        return outputs.logits, outputs.loss
