"""
Training script for Perceiver IO + Qwen 3 4B model.
"""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

from model import PerceiverQwenModel
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
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--latent_dim", type=int, default=512, help="Perceiver IO latent dimension")
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
    
    # Initialize model
    print("Initializing model...")
    model = PerceiverQwenModel(
        qwen_model_name=args.qwen_model_name,
        perceiver_model_name=args.perceiver_model_name,
        latent_dim=args.latent_dim,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.train()
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler
    num_training_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
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
            turn_boundaries = batch["turn_boundaries"]
            
            # Forward pass
            optimizer.zero_grad()
            
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                turn_boundaries=turn_boundaries,
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": epoch_loss / global_step})
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model
                model.qwen_model.save_pretrained(checkpoint_dir)
                
                # Save other components
                torch.save({
                    "compressor": model.compressor.state_dict(),
                    "projection": model.projection.state_dict(),
                    "perceiver": model.perceiver.state_dict(),
                }, os.path.join(checkpoint_dir, "additional_components.pt"))
                
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
    torch.save({
        "compressor": model.compressor.state_dict(),
        "projection": model.projection.state_dict(),
        "perceiver": model.perceiver.state_dict(),
    }, os.path.join(final_dir, "additional_components.pt"))
    tokenizer.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)

