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
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        # Load base Qwen model (using AutoModelForCausalLM for compatibility)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.float16,
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
        
        # Get model dtype and ensure Perceiver model and input projection match
        model_dtype = next(self.qwen_model.parameters()).dtype
        
        # Convert Perceiver model to match Qwen model dtype
        if self.perceiver.perceiver is not None:
            self.perceiver.perceiver = self.perceiver.perceiver.to(dtype=model_dtype)
            print(f"Converted Perceiver model to dtype: {model_dtype}")
        
        # Ensure input projection matches dtype
        if self.perceiver.input_projection is not None:
            self.perceiver.input_projection = self.perceiver.input_projection.to(dtype=model_dtype)
            print(f"Set input projection dtype to: {model_dtype}")
        
        # Freeze base PerceiverModel (compressor and projection remain trainable)
        self.perceiver.freeze_base_model()
        
        # Cross-attention compressor (trainable)
        self.compressor = CrossAttentionCompressor(
            latent_dim=latent_dim,
            num_heads=8,
        )
        
        # Projection layer from Perceiver latent space to LLM embedding space (trainable)
        self.projection = nn.Linear(latent_dim, self.llm_embed_dim)
        
        # Get model dtype and ensure all layers match
        model_dtype = next(self.qwen_model.parameters()).dtype
        
        # Convert compressor to match model dtype
        self.compressor = self.compressor.to(dtype=model_dtype)
        print(f"Converted compressor to dtype: {model_dtype}")
        
        # Convert projection layer to match model dtype
        self.projection = self.projection.to(dtype=model_dtype)
        print(f"Converted projection layer to dtype: {model_dtype}")
        
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
        
        # Enable gradient checkpointing for memory efficiency
        # This trades compute for memory by recomputing activations during backward pass
        if gradient_checkpointing:
            if hasattr(self.qwen_model, 'gradient_checkpointing_enable'):
                self.qwen_model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled for Qwen model")
            elif hasattr(self.qwen_model, 'model') and hasattr(self.qwen_model.model, 'gradient_checkpointing_enable'):
                # For PEFT models, the base model might be in .model
                self.qwen_model.model.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled for Qwen base model")
            else:
                print("Warning: Gradient checkpointing not available for this model")
        else:
            print("Gradient checkpointing disabled")
            
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
        if turn_boundaries is None or len(turn_boundaries) == 0:
            # If no turn boundaries, treat entire conversation as last turn
            # This handles edge cases where no turns are detected
            return [], conversation_ids
        
        # Handle case where turn_boundaries might be a list of lists (batch dimension)
        # If first element is a list, extract it (assuming batch_size=1 for now)
        if len(turn_boundaries) > 0 and isinstance(turn_boundaries[0], list):
            turn_boundaries = turn_boundaries[0]
        
        # Check again after extraction
        if len(turn_boundaries) == 0:
            return [], conversation_ids
        
        turns = []
        for start, end in turn_boundaries[:-1]:
            turns.append(conversation_ids[:, start:end])
        
        # Last turn - truncate to last 100 tokens to reduce memory usage
        last_start, last_end = turn_boundaries[-1]
        max_last_turn_length = 100
        turn_length = last_end - last_start
        if turn_length > max_last_turn_length:
            # Take the last 100 tokens
            actual_start = last_end - max_last_turn_length
            last_turn = conversation_ids[:, actual_start:last_end]
        else:
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
        
        # Get model dtype to ensure consistency
        model_dtype = next(self.qwen_model.parameters()).dtype
        
        for turn in turns:
            # Convert token IDs to embeddings
            # Perceiver IO can accept embeddings as input
            embed_layer = self.qwen_model.get_input_embeddings()
            turn_embeddings = embed_layer(turn)
            # Ensure dtype matches model
            if turn_embeddings.dtype != model_dtype:
                turn_embeddings = turn_embeddings.to(dtype=model_dtype)
            
            # Process through Perceiver IO
            # Perceiver expects inputs of shape (batch, seq_len, input_dim)
            latents = self.perceiver(turn_embeddings)
            # Ensure output dtype matches model
            if latents.dtype != model_dtype:
                latents = latents.to(dtype=model_dtype)
            all_latents.append(latents)
        
        # Concatenate all latents from different turns
        # Shape: (batch_size, num_turns * num_latents, latent_dim)
        if all_latents:
            combined_latents = torch.cat(all_latents, dim=1)
        else:
            # If no turns, create empty latents
            # Create a dummy input that will flow through the trainable compressor
            batch_size = turns[0].shape[0] if turns else 1
            device = next(self.qwen_model.parameters()).device
            model_dtype = next(self.qwen_model.parameters()).dtype
            # Use compressor's query as base to ensure gradient flow
            combined_latents = self.compressor.query.expand(batch_size, 1, -1) * 0
        
        # Compress to single vector using cross-attention
        # Even if combined_latents comes from frozen Perceiver, compressor is trainable
        # so gradients will flow through compressor parameters
        compressed = self.compressor(combined_latents)
        
        # Ensure output dtype matches model
        model_dtype = next(self.qwen_model.parameters()).dtype
        if compressed.dtype != model_dtype:
            compressed = compressed.to(dtype=model_dtype)
        
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
        # Handle case where turn_boundaries might be a list of lists (batch dimension)
        # Extract boundaries for the first batch item if needed
        if turn_boundaries is not None and len(turn_boundaries) > 0 and isinstance(turn_boundaries[0], list):
            turn_boundaries = turn_boundaries[0]
        
        # Split conversation into turns
        turns, last_turn = self.split_conversation_turns(input_ids, turn_boundaries)
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Get model dtype to ensure all tensors match
        model_dtype = next(self.qwen_model.parameters()).dtype
        
        # Process turns through Perceiver IO
        if turns:
            compressed_latent = self.encode_turns_with_perceiver(turns)
            # Project to LLM embedding space
            projected_embedding = self.projection(compressed_latent)  # (batch_size, 1, llm_embed_dim)
            # Ensure dtype matches model
            if projected_embedding.dtype != model_dtype:
                projected_embedding = projected_embedding.to(dtype=model_dtype)
        else:
            # If no previous turns, create zero embedding through projection layer
            # This ensures gradients can flow through the projection layer
            zero_latent = torch.zeros(
                batch_size, 
                1, 
                self.projection.in_features,  # latent_dim
                device=device,
                dtype=model_dtype
            )
            projected_embedding = self.projection(zero_latent)
        
        # Get embeddings for last turn
        embed_layer = self.qwen_model.get_input_embeddings()
        last_turn_embeddings = embed_layer(last_turn)
        # Ensure dtype matches model
        if last_turn_embeddings.dtype != model_dtype:
            last_turn_embeddings = last_turn_embeddings.to(dtype=model_dtype)
        
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
            last_start, last_end = turn_boundaries[-1] if turn_boundaries and len(turn_boundaries) > 0 else (0, last_turn.shape[1])
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
            last_start, last_end = turn_boundaries[-1] if turn_boundaries and len(turn_boundaries) > 0 else (0, last_turn.shape[1])
            
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

