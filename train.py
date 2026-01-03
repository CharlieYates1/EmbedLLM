"""
Training script for Perceiver IO + Qwen 3 4B model.
"""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from accelerate import Accelerator

from QwenWithPerceiverCrossAttn import QwenWithPerceiverCrossAttn
from data_utils import ConversationDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train Perceiver IO + Qwen 3 4B model")
    parser.add_argument("--data_path", type=str, default="Bossologist/general_Qwen3_ft_dataset",
                        help="Path to conversation data JSON or HuggingFace dataset name")
    parser.add_argument("--text_column", type=str, default=None,
                        help="Column name in HuggingFace dataset containing conversation text (auto-detected if not specified)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--qwen_model_name", type=str, default="Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged", 
                        help="Qwen model name")
    parser.add_argument("--perceiver_model_name", type=str, default="deepmind/multimodal-perceiver",
                        help="Perceiver IO model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--latent_dim", type=int, default=512, help="Perceiver IO latent dimension")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save memory (default: True)")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false",
                        help="Disable gradient checkpointing")
    return parser.parse_args()


def train(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("Loading dataset...")
    dataset = ConversationDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_column=args.text_column,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    # Only keep at most 10000 batches in the DataLoader by subsampling the dataset
    max_batches = 7000
    batches_in_dataset = len(dataloader)
    if batches_in_dataset > max_batches:
        print(f"Limiting DataLoader to {max_batches} batches (original: {batches_in_dataset})")
        # Convert DataLoader to list, subsample, then reload as DataLoader for batching
        sampled_indices = torch.randperm(len(dataset))[:args.batch_size * max_batches].tolist()
        # Create a new dataset with only sampled indices
        from torch.utils.data import Subset
        dataset = Subset(dataset, sampled_indices)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
    # Initialize model
    print("Initializing model...")
    model = QwenWithPerceiverCrossAttn(
        qwen_model_name=args.qwen_model_name,
        perceiver_model_name=args.perceiver_model_name,
    )
    model.train()
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    # Optimizer - only optimize trainable parameters to save memory
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizer tracking {len(trainable_params)} parameter groups (trainable only)")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=0.01,
        eps=1e-4,
    )
    
    # Learning rate scheduler
    num_training_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    accelerator = Accelerator()
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # Training loop
    print("Starting training...")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if global_step == 0:
                
                print(f"Input shape: {input_ids.shape}, Labels shape: {labels.shape}")
                
                # Show first few tokens of input
                input_sample = input_ids[0, :50].cpu().tolist()  # First 50 tokens
                print(f"First 50 input tokens: {input_sample}")
                print(f"Input text (First 200 chars): {tokenizer.decode(input_ids[0], skip_special_tokens=True)[:200]}")
                print("=" * 50 + "\n")

                print(f"First 50 labels: {labels[0, :50].cpu().tolist()}")
                print("=" * 50 + "\n")
            
            # Forward pass
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                perceiver_input_ids=input_ids,
            )

            if torch.isinf(logits).any() or torch.isnan(logits).any():
                print("WARNING: Detected NaN or Inf in logits at global_step", global_step)
                print("Logits:", logits)
                continue

            if global_step == 0:
                print("Predicted text: ", tokenizer.decode(torch.argmax(logits, dim=-1)[0][:10], skip_special_tokens=True))
            
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(), 
                "avg_loss": epoch_loss / global_step,
                "grad_norm": grad_norm.item() if not torch.isnan(grad_norm) else 0.0
            })
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model
                model.qwen_model.save_pretrained(checkpoint_dir)
                
                # Save Perceiver module and cross-attention components
                additional_components = {
                    "perceiver": model.perceiver.state_dict(),
                }
                
                # Save cross-attention modules from the modified layer
                if hasattr(model.qwen_model, 'model') and hasattr(model.qwen_model.model, 'layers'):
                    layers = model.qwen_model.model.layers
                elif hasattr(model.qwen_model, 'layers'):
                    layers = model.qwen_model.layers
                else:
                    layers = None
                
                if layers is not None and model.layer_index < len(layers):
                    target_layer = layers[model.layer_index]
                    if hasattr(target_layer, 'perceiver_cross_attn'):
                        additional_components["perceiver_cross_attn"] = target_layer.perceiver_cross_attn.state_dict()
                    if hasattr(target_layer, 'cross_attn_layer_norm'):
                        additional_components["cross_attn_layer_norm"] = target_layer.cross_attn_layer_norm.state_dict()
                
                torch.save(additional_components, os.path.join(checkpoint_dir, "additional_components.pt"))
                
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"\nSaved checkpoint to {checkpoint_dir}")
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    print("Saving final model...")
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.qwen_model.save_pretrained(final_dir)
    
    # Save Perceiver module and cross-attention components
    additional_components = {
        "perceiver": model.perceiver.state_dict(),
    }
    
    # Save cross-attention modules from the modified layer
    if hasattr(model.qwen_model, 'model') and hasattr(model.qwen_model.model, 'layers'):
        layers = model.qwen_model.model.layers
    elif hasattr(model.qwen_model, 'layers'):
        layers = model.qwen_model.layers
    else:
        layers = None
    
    if layers is not None and model.layer_index < len(layers):
        target_layer = layers[model.layer_index]
        if hasattr(target_layer, 'perceiver_cross_attn'):
            additional_components["perceiver_cross_attn"] = target_layer.perceiver_cross_attn.state_dict()
        if hasattr(target_layer, 'cross_attn_layer_norm'):
            additional_components["cross_attn_layer_norm"] = target_layer.cross_attn_layer_norm.state_dict()
    
    torch.save(additional_components, os.path.join(final_dir, "additional_components.pt"))
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)

