"""
Qwen model with cross-attention to Perceiver outputs at layer 8.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Optional, Tuple
from perceiver_module import PerceiverIOModule
from PerceiverCrossAttention import PerceiverCrossAttention


class QwenWithPerceiverCrossAttn(nn.Module):
    """
    Qwen model with cross-attention to Perceiver outputs at layer 8.
    
    Architecture:
    1. Process conversation turns (except last) through Perceiver IO to get latent representations
    2. Feed full input sequence through Qwen model layers
    3. At layer 8, inject cross-attention where queries come from layer 8 outputs
       and keys/values come from Perceiver outputs
    4. Continue through remaining Qwen layers with the cross-attention enhanced hidden states
    """
    def __init__(
        self,
        qwen_model_name: str = "Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged",
        perceiver_model_name: str = "deepmind/multimodal-perceiver",
        insert_cross_attn_at_layer: int = 8,
        dropout: float = 0.0,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.insert_cross_attn_at_layer = insert_cross_attn_at_layer
        
        # Load base Qwen model
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # Get model configuration
        self.hidden_size = self.qwen_model.config.hidden_size
        
        # Get model dtype
        model_dtype = next(self.qwen_model.parameters()).dtype
        
        # Perceiver IO module for processing conversation turns
        self.perceiver = PerceiverIOModule(
            model_name=perceiver_model_name,
            input_dim=self.hidden_size,  # Perceiver receives LLM embeddings
        ).to(dtype=model_dtype)
        
        # Create cross-attention block
        self.cross_attention = PerceiverCrossAttention(
            hidden_size=self.hidden_size,
            perceiver_dim=self.perceiver.latent_dim,  # After projection
            num_heads=self.qwen_model.config.num_attention_heads,
            dropout=dropout,
        ).to(dtype=model_dtype)
        
        # Store reference to original forward method
        self._original_forward = self.qwen_model.forward
        
        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.qwen_model = get_peft_model(self.qwen_model, lora_config)
        
        # Enable gradient checkpointing
        if gradient_checkpointing:
            if hasattr(self.qwen_model, 'gradient_checkpointing_enable'):
                self.qwen_model.gradient_checkpointing_enable()
            elif hasattr(self.qwen_model, 'model') and hasattr(self.qwen_model.model, 'gradient_checkpointing_enable'):
                self.qwen_model.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        self._modify_target_layer()

    def _modify_target_layer(self):
        qwen_layers = self.qwen_model.model.model.layers
        target_layer = qwen_layers[self.insert_cross_attn_at_layer]
        target_layer.perceiver_cross_attention = self.cross_attention
        target_layer.cross_attn_layer_norm = nn.LayerNorm(self.hidden_size)
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
            perceiver_outputs=None,
            **kwargs,
        ):
            hidden_states = original_forward(
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
            
            if perceiver_outputs is not None:
                cross_attn_output, _ = target_layer.perceiver_cross_attention(
                    hidden_states=hidden_states,
                    perceiver_outputs=perceiver_outputs,
                )
                hidden_states = target_layer.cross_attn_layer_norm(
                    hidden_states + cross_attn_output
                )
            
            return hidden_states
        
        target_layer.forward = patched_forward
        
        print(f"Cross-attention inserted at layer {self.insert_cross_attn_at_layer}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the combined model.
        """
        # Process full input sequence through Perceiver IO
        embed_layer = self.qwen_model.get_input_embeddings()
        turn_embeddings = embed_layer(input_ids)
        perceiver_outputs = self.perceiver(turn_embeddings)
        
        # Process full input sequence through Qwen model
        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            perceiver_outputs=perceiver_outputs,
        )
        
        return outputs.logits, outputs.loss

