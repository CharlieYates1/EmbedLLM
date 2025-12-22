"""
Custom model architecture combining Perceiver IO with Qwen 3 4B.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import List, Optional, Tuple
from perceiver_module import PerceiverIOModule, CrossAttentionCompressor


class PerceiverQwenModel(nn.Module):
    """
    Model that integrates Perceiver IO with Qwen 3 4B for conversation understanding.
    
    Architecture:
    1. Process conversation turns (except last) through Perceiver IO
    2. Compress Perceiver IO latents to single vector via cross-attention
    3. Project to LLM embedding space
    4. Combine with last turn tokens
    5. Feed through Qwen LLM for next token prediction
    """
    def __init__(
        self,
        qwen_model_name: str = "Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged",
        perceiver_model_name: str = "deepmind/multimodal-perceiver",
        latent_dim: int = 512,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Load base Qwen model (using AutoModelForCausalLM for compatibility)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Get LLM embedding dimension
        self.llm_embed_dim = self.qwen_model.config.hidden_size
        
        # Perceiver IO module for processing conversation turns
        self.perceiver = PerceiverIOModule(
            model_name=perceiver_model_name,
            latent_dim=latent_dim,
            input_dim=self.llm_embed_dim,  # Perceiver will receive LLM embeddings
        )
        
        # Cross-attention compressor
        self.compressor = CrossAttentionCompressor(
            latent_dim=latent_dim,
            num_heads=8,
        )
        
        # Projection layer from Perceiver latent space to LLM embedding space
        self.projection = nn.Linear(latent_dim, self.llm_embed_dim)
        
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
            
    def split_conversation_turns(
        self, 
        conversation_ids: torch.Tensor,
        turn_boundaries: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Split conversation into turns. Last turn is returned separately.
        
        Args:
            conversation_ids: Token IDs of the entire conversation
            turn_boundaries: Optional list of (start, end) indices for each turn
            
        Returns:
            Tuple of (list of turn tensors except last, last turn tensor)
        """
        # If turn boundaries not provided, assume conversation is already split
        # This is a placeholder - implement based on your data format
        if turn_boundaries is None:
            # Simple split: assume turns are separated by special tokens or equal lengths
            # This needs to be customized based on your data format
            raise NotImplementedError("Turn boundary detection needs to be implemented based on your data format")
        
        turns = []
        for start, end in turn_boundaries[:-1]:
            turns.append(conversation_ids[:, start:end])
        
        # Last turn
        last_start, last_end = turn_boundaries[-1]
        last_turn = conversation_ids[:, last_start:last_end]
        
        return turns, last_turn
    
    def encode_turns_with_perceiver(
        self,
        turns: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Process conversation turns through Perceiver IO.
        
        Args:
            turns: List of turn tensors (each is batch_size x seq_len with token IDs)
            
        Returns:
            Compressed vector from Perceiver IO of shape (batch_size, 1, latent_dim)
        """
        all_latents = []
        
        for turn in turns:
            # Convert token IDs to embeddings
            # Perceiver IO can accept embeddings as input
            turn_embeddings = self.qwen_model.model.embed_tokens(turn)
            
            # Process through Perceiver IO
            # Perceiver expects inputs of shape (batch, seq_len, input_dim)
            latents = self.perceiver(turn_embeddings)
            all_latents.append(latents)
        
        # Concatenate all latents from different turns
        # Shape: (batch_size, num_turns * num_latents, latent_dim)
        if all_latents:
            combined_latents = torch.cat(all_latents, dim=1)
        else:
            # If no turns, create empty latents
            batch_size = turns[0].shape[0] if turns else 1
            device = next(self.perceiver.parameters()).device
            dtype = next(self.perceiver.parameters()).dtype
            combined_latents = torch.zeros(
                batch_size, 
                1, 
                self.perceiver.latent_dim,
                device=device,
                dtype=dtype
            )
        
        # Compress to single vector using cross-attention
        compressed = self.compressor(combined_latents)
        
        return compressed
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        turn_boundaries: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the combined model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask
            labels: Labels for next token prediction
            turn_boundaries: Boundaries of conversation turns
            tokenizer: Tokenizer for processing
            
        Returns:
            Tuple of (logits, loss)
        """
        # Split conversation into turns
        turns, last_turn = self.split_conversation_turns(input_ids, turn_boundaries)
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Process turns through Perceiver IO
        if turns:
            compressed_latent = self.encode_turns_with_perceiver(turns)
            # Project to LLM embedding space
            projected_embedding = self.projection(compressed_latent)  # (batch_size, 1, llm_embed_dim)
        else:
            # If no previous turns, create zero embedding
            dtype = next(self.projection.parameters()).dtype
            projected_embedding = torch.zeros(
                batch_size, 
                1, 
                self.llm_embed_dim,
                device=device,
                dtype=dtype
            )
        
        # Get embeddings for last turn
        last_turn_embeddings = self.qwen_model.model.embed_tokens(last_turn)
        
        # Concatenate: [compressed_context, last_turn_tokens]
        combined_embeddings = torch.cat([projected_embedding, last_turn_embeddings], dim=1)
        
        # Create attention mask for combined embeddings
        if attention_mask is not None:
            # Add 1 for the compressed context token
            context_mask = torch.ones(
                batch_size, 
                1, 
                device=device,
                dtype=attention_mask.dtype
            )
            # Get mask for last turn only
            last_start, last_end = turn_boundaries[-1] if turn_boundaries else (0, last_turn.shape[1])
            last_turn_mask = attention_mask[:, last_start:last_end]
            combined_mask = torch.cat([context_mask, last_turn_mask], dim=1)
        else:
            combined_mask = torch.ones(
                batch_size,
                combined_embeddings.shape[1],
                device=device,
                dtype=torch.long
            )
        
        # Adjust labels to match the new sequence length
        # Labels should be aligned with the combined embeddings for next token prediction
        if labels is not None:
            # For next token prediction, labels[i] should predict token at position i+1
            # When we add a context token, labels need to be shifted
            
            # Context token doesn't predict anything (use -100 to ignore in loss)
            context_labels = torch.full(
                (batch_size, 1),
                -100,
                device=device,
                dtype=labels.dtype
            )
            
            # Get labels for the last turn
            # Labels are already shifted by 1 in data_utils (labels[i] corresponds to next token)
            last_start, last_end = turn_boundaries[-1] if turn_boundaries else (0, last_turn.shape[1])
            
            # Extract corresponding labels for last turn
            # labels[start:end] correspond to predicting tokens at [start+1:end+1]
            if last_end < labels.shape[1]:
                # Get labels for positions [last_start:last_end-1] to align with last_turn tokens
                last_turn_labels = labels[:, last_start:min(last_end, labels.shape[1])]
            elif last_start < labels.shape[1]:
                # Partial overlap
                last_turn_labels = labels[:, last_start:]
            else:
                # No overlap
                last_turn_labels = torch.full(
                    (batch_size, 1),
                    -100,
                    device=device,
                    dtype=labels.dtype
                )
            
            # Ensure labels match the length of last_turn
            if last_turn_labels.shape[1] < last_turn.shape[1]:
                # Pad with -100
                pad_size = last_turn.shape[1] - last_turn_labels.shape[1]
                padding = torch.full(
                    (batch_size, pad_size),
                    -100,
                    device=device,
                    dtype=labels.dtype
                )
                last_turn_labels = torch.cat([last_turn_labels, padding], dim=1)
            elif last_turn_labels.shape[1] > last_turn.shape[1]:
                # Truncate
                last_turn_labels = last_turn_labels[:, :last_turn.shape[1]]
            
            combined_labels = torch.cat([context_labels, last_turn_labels], dim=1)
        else:
            combined_labels = None
        
        # Forward through Qwen LLM
        # We use inputs_embeds since we already have embeddings
        outputs = self.qwen_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            labels=combined_labels,
        )
        
        return outputs.logits, outputs.loss

