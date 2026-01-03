"""
Utility function to load model from checkpoint.
"""
import torch
import os
from transformers import AutoTokenizer
from QwenWithPerceiverCrossAttn import QwenWithPerceiverCrossAttn


def load_model_from_checkpoint(
    checkpoint_dir: str,
    qwen_model_name: str = "Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged",
    perceiver_model_name: str = "deepmind/multimodal-perceiver",
    device: str = None,
):
    """
    Load QwenWithPerceiverCrossAttn model from checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        qwen_model_name: Base Qwen model name (used if checkpoint doesn't have config)
        perceiver_model_name: Perceiver model name
        device: Device to load model on (default: cuda if available, else cpu)
    
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from checkpoint: {checkpoint_dir}")
    
    # Load tokenizer
    tokenizer_path = checkpoint_dir
    if os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json")):
        print("Loading tokenizer from checkpoint...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print(f"Tokenizer not found in checkpoint, loading from {qwen_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if checkpoint has model files, if so use checkpoint, otherwise use base model name
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        print("Found model config in checkpoint, using checkpoint as model source...")
        model_source = checkpoint_dir
    else:
        print(f"No model config in checkpoint, using base model: {qwen_model_name}")
        model_source = qwen_model_name
    
    # Initialize model structure (will load weights from checkpoint if available)
    print("Initializing model structure...")
    model = QwenWithPerceiverCrossAttn(
        qwen_model_name=model_source,  # Use checkpoint if available, else base model
        perceiver_model_name=perceiver_model_name,
    )
    
    # If we used the checkpoint as source, the weights are already loaded
    # Otherwise, the base model weights are loaded (which is fine for resuming)
    print("Model structure initialized")
    
    # Load additional components (perceiver, cross-attention)
    additional_path = os.path.join(checkpoint_dir, "additional_components.pt")
    if os.path.exists(additional_path):
        print("Loading additional components from checkpoint...")
        additional_components = torch.load(additional_path, map_location=device)
        
        # Load perceiver weights
        if "perceiver" in additional_components:
            try:
                model.perceiver.load_state_dict(additional_components["perceiver"], strict=False)
                print("Loaded Perceiver weights")
            except Exception as e:
                print(f"Warning: Could not load Perceiver weights: {e}")
        
        # Load cross-attention modules
        if hasattr(model.qwen_model, 'model') and hasattr(model.qwen_model.model, 'layers'):
            layers = model.qwen_model.model.layers
        elif hasattr(model.qwen_model, 'layers'):
            layers = model.qwen_model.layers
        else:
            layers = None
        
        if layers is not None and model.layer_index < len(layers):
            target_layer = layers[model.layer_index]
            
            if "perceiver_cross_attn" in additional_components:
                if hasattr(target_layer, 'perceiver_cross_attn'):
                    try:
                        target_layer.perceiver_cross_attn.load_state_dict(
                            additional_components["perceiver_cross_attn"], strict=False
                        )
                        print("Loaded PerceiverCrossAttention weights")
                    except Exception as e:
                        print(f"Warning: Could not load PerceiverCrossAttention weights: {e}")
            
            if "cross_attn_layer_norm" in additional_components:
                if hasattr(target_layer, 'cross_attn_layer_norm'):
                    try:
                        target_layer.cross_attn_layer_norm.load_state_dict(
                            additional_components["cross_attn_layer_norm"], strict=False
                        )
                        print("Loaded CrossAttention LayerNorm weights")
                    except Exception as e:
                        print(f"Warning: Could not load CrossAttention LayerNorm weights: {e}")
    else:
        print("No additional_components.pt found in checkpoint (this is okay for initial checkpoints)")
    
    # Move model to device
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    
    return model, tokenizer


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
        model, tokenizer = load_model_from_checkpoint(checkpoint_dir)
        print(f"\nModel loaded successfully!")
        print(f"Model type: {type(model)}")
    else:
        print("Usage: python load_checkpoint.py <checkpoint_dir>")

