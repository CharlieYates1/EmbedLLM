# EmbedLLM: Perceiver IO + Qwen 3 4B Fine-tuning

This project fine-tunes Qwen 3 4B (2507) using LoRA to integrate Perceiver IO for processing conversation context. The model processes conversation turns through Perceiver IO, converts the latent space to LLM token embeddings, and predicts the next token.

## Architecture

1. **Conversation Processing**: Splits conversation into turns
2. **Perceiver IO Encoding**: Processes all turns except the last through Perceiver IO
3. **Latent Space Compression**: Uses cross-attention to condense Perceiver IO outputs into a single vector
4. **Embedding Projection**: Linear layer projects the compressed vector to LLM embedding space
5. **Token Prediction**: Combined embeddings + last turn tokens are fed into Qwen 3 4B for next token prediction

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Using HuggingFace Dataset (default)

The default dataset is `Bossologist/general_Qwen3_ft_dataset`. Simply run:

```bash
python train.py --output_dir ./checkpoints
```

### Using Local JSON File

1. Prepare your conversation data in JSON format (see `sample_data.json` for example):
```json
[
    {
        "conversation": "turn1<|turn|>turn2<|turn|>turn3"
    }
]
```

2. Run training:
```bash
python train.py --data_path your_conversation_data.json --output_dir ./checkpoints
```

### Options

- `--data_path`: Path to local JSON file or HuggingFace dataset name (default: `Bossologist/general_Qwen3_ft_dataset`)
- `--text_column`: Column name in HuggingFace dataset containing conversation text (auto-detected if not specified)
- `--qwen_model_name`: Model name (default: `Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged`)

## Configuration

Key parameters you can adjust:
- `--qwen_model_name`: Qwen model to use (default uses smaller model for testing)
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha scaling (default: 32)
- `--batch_size`: Training batch size (default: 4)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--latent_dim`: Perceiver IO latent dimension (default: 512)

## Model Files

- `model.py`: Custom model architecture integrating Perceiver IO and Qwen 3 4B
- `perceiver_module.py`: Perceiver IO integration utilities
- `train.py`: Training script with LoRA configuration
- `data_utils.py`: Conversation data preprocessing

