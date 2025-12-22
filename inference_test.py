"""
Inference script to test the PerceiverQwenModel with dummy conversations.
"""
import torch
import os
from transformers import AutoTokenizer
from model import PerceiverQwenModel
from data_utils import ConversationDataset


def find_turn_boundaries(token_ids, turn_separator_ids):
    """Find turn boundaries in tokenized conversation."""
    boundaries = []
    start = 0
    i = 0
    
    while i < len(token_ids):
        if i + len(turn_separator_ids) <= len(token_ids):
            if token_ids[i:i+len(turn_separator_ids)] == turn_separator_ids:
                if start < i:
                    boundaries.append((start, i))
                start = i + len(turn_separator_ids)
                i = start
                continue
        i += 1
    
    if start < len(token_ids):
        boundaries.append((start, len(token_ids)))
    
    return boundaries if boundaries else [(0, len(token_ids))]


def load_model_from_checkpoint(checkpoint_dir, device):
    """Load model from a training checkpoint."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    
    print(f"Loading model from checkpoint: {checkpoint_dir}")
    
    # Load base model
    qwen_model_name = "Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged"
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Check if it's a LoRA checkpoint
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        base_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        print("Loaded LoRA adapters from checkpoint")
    
    # Load additional components if they exist
    additional_path = os.path.join(checkpoint_dir, "additional_components.pt")
    if os.path.exists(additional_path):
        print("Found additional components, but need to rebuild full model structure")
        # Note: This would require reconstructing the full PerceiverQwenModel
        # For now, we'll use the base model
    
    return base_model


def test_inference(checkpoint_dir=None):
    """Test model inference with dummy conversation."""
    print("=" * 60)
    print("Testing PerceiverQwenModel Inference")
    print("=" * 60)
    
    # Model configuration
    qwen_model_name = "Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged"
    perceiver_model_name = "deepmind/multimodal-perceiver"
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer_path = checkpoint_dir if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json")) else qwen_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    # Load model
    print("\n[2/5] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        print(f"Loading from checkpoint: {checkpoint_dir}")
        # For now, load base model and reconstruct PerceiverQwenModel
        # In practice, you'd want to save/load the full model structure
        model = PerceiverQwenModel(
            qwen_model_name=checkpoint_dir if os.path.exists(os.path.join(checkpoint_dir, "config.json")) else qwen_model_name,
            perceiver_model_name=perceiver_model_name,
            use_lora=True,  # Assume checkpoint uses LoRA
        )
        # Try to load additional components
        additional_path = os.path.join(checkpoint_dir, "additional_components.pt")
        if os.path.exists(additional_path):
            additional = torch.load(additional_path, map_location=device)
            model.compressor.load_state_dict(additional["compressor"])
            model.projection.load_state_dict(additional["projection"])
            model.perceiver.load_state_dict(additional["perceiver"])
            print("Loaded additional components (compressor, projection, perceiver)")
    else:
        print("Loading base model (no checkpoint)")
        model = PerceiverQwenModel(
            qwen_model_name=qwen_model_name,
            perceiver_model_name=perceiver_model_name,
            use_lora=False,  # For testing base model
        )
    
    model.eval()
    model = model.to(device)
    print("Model loaded successfully!")
    
    # Create dummy conversation
    print("\n[3/5] Preparing dummy conversation...")
    turn_separator = "<|turn|>"
    dummy_conversation = (
        "Hello, how are you today?"
        f"{turn_separator}"
        "I'm doing great, thanks for asking! How about you?"
        f"{turn_separator}"
        "I'm also doing well. What are you working on?"
        f"{turn_separator}"
        "I'm working on a machine learning project."
    )
    
    print(f"Conversation:\n{dummy_conversation}")
    print(f"\nNumber of turns: {dummy_conversation.count(turn_separator) + 1}")
    
    # Tokenize conversation
    print("\n[4/5] Tokenizing conversation...")
    encoded = tokenizer(
        dummy_conversation,
        max_length=2048,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # Find turn boundaries
    token_ids_list = input_ids[0].tolist()
    turn_separator_ids = tokenizer.encode(turn_separator, add_special_tokens=False)
    turn_boundaries = find_turn_boundaries(token_ids_list, turn_separator_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Turn boundaries: {turn_boundaries}")
    print(f"Number of turns: {len(turn_boundaries)}")
    
    # Run inference
    print("\n[5/5] Running inference...")
    with torch.no_grad():
        logits, loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            turn_boundaries=[turn_boundaries],  # Batch dimension
        )
    
    print(f"Output logits shape: {logits.shape}")
    if loss is not None:
        print(f"Loss: {loss.item():.4f}")
    
    # Get predicted tokens
    predicted_token_ids = torch.argmax(logits, dim=-1)
    
    # Decode predictions (for the last few positions)
    print("\n" + "=" * 60)
    print("Predictions (last 20 tokens):")
    print("=" * 60)
    
    last_n = min(20, predicted_token_ids.shape[1])
    predicted_text = tokenizer.decode(
        predicted_token_ids[0, -last_n:],
        skip_special_tokens=True
    )
    actual_text = tokenizer.decode(
        input_ids[0, -last_n:],
        skip_special_tokens=True
    )
    
    print(f"\nActual tokens (last {last_n}):")
    print(actual_text)
    print(f"\nPredicted tokens (last {last_n}):")
    print(predicted_text)
    
    # Show token-level predictions for a few positions
    print("\n" + "=" * 60)
    print("Token-level predictions (last 10 positions):")
    print("=" * 60)
    for i in range(max(0, predicted_token_ids.shape[1] - 10), predicted_token_ids.shape[1]):
        actual_token_id = input_ids[0, i].item()
        predicted_token_id = predicted_token_ids[0, i].item()
        actual_token = tokenizer.decode([actual_token_id], skip_special_tokens=False)
        predicted_token = tokenizer.decode([predicted_token_id], skip_special_tokens=False)
        
        match = "✓" if actual_token_id == predicted_token_id else "✗"
        print(f"Position {i:3d}: Actual='{actual_token:15s}' Predicted='{predicted_token:15s}' {match}")
    
    print("\n" + "=" * 60)
    print("Inference test completed!")
    print("=" * 60)


def test_generation():
    """Test text generation with the model."""
    print("\n" + "=" * 60)
    print("Testing Text Generation")
    print("=" * 60)
    
    qwen_model_name = "Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged"
    perceiver_model_name = "deepmind/multimodal-perceiver"
    
    # Load tokenizer
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerceiverQwenModel(
        qwen_model_name=qwen_model_name,
        perceiver_model_name=perceiver_model_name,
        use_lora=False,
    )
    model.eval()
    model = model.to(device)
    
    # Create conversation context
    turn_separator = "<|turn|>"
    conversation = (
        "What is machine learning?"
        f"{turn_separator}"
        "Machine learning is a subset of artificial intelligence."
        f"{turn_separator}"
        "Can you give me an example?"
        f"{turn_separator}"
    )
    
    print(f"\nConversation context:\n{conversation}")
    
    # Tokenize
    encoded = tokenizer(
        conversation,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    
    # Find turn boundaries
    token_ids_list = input_ids[0].tolist()
    turn_separator_ids = tokenizer.encode(turn_separator, add_special_tokens=False)
    turn_boundaries = find_turn_boundaries(token_ids_list, turn_separator_ids)
    
    print(f"\nTurn boundaries: {turn_boundaries}")
    print("\nGenerating response...")
    
    # Use the underlying Qwen model for generation
    # Note: This is a simplified generation - you may want to use the full model pipeline
    with torch.no_grad():
        logits, _ = model(
            input_ids=input_ids,
            turn_boundaries=[turn_boundaries],
        )
    
    # Sample next token (greedy decoding)
    next_token_id = torch.argmax(logits[0, -1, :], dim=-1)
    next_token = tokenizer.decode([next_token_id], skip_special_tokens=False)
    
    print(f"\nGenerated next token: '{next_token}'")
    print(f"Full predicted response start: {tokenizer.decode([next_token_id], skip_special_tokens=True)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PerceiverQwenModel inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint directory (optional)")
    args = parser.parse_args()
    
    try:
        test_inference(checkpoint_dir=args.checkpoint)
        # Uncomment to test generation
        # test_generation()
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
