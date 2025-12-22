"""
Data utilities for processing conversation data.
"""
import json
import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not available. Install with: pip install datasets")


class ConversationDataset(Dataset):
    """
    Dataset for conversation data with turn boundaries.
    Supports both local JSON files and HuggingFace datasets.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        turn_separator: str = "<|im_start|>",
        text_column: Optional[str] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSON file with conversations OR HuggingFace dataset name
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            turn_separator: Token/string used to separate turns
            text_column: Column name in HuggingFace dataset containing conversation text
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.turn_separator = turn_separator
        
        # Check if data_path is a HuggingFace dataset or local file
        if "/" in data_path and not os.path.exists(data_path) and DATASETS_AVAILABLE:
            # Likely a HuggingFace dataset name
            print(f"Loading dataset from HuggingFace: {data_path}")
            try:
                hf_dataset = load_dataset(data_path)
                # Handle different dataset splits
                if isinstance(hf_dataset, dict):
                    # Multiple splits, use 'train' by default
                    if 'train' in hf_dataset:
                        dataset_split = hf_dataset['train']
                    else:
                        # Use first available split
                        dataset_split = list(hf_dataset.values())[0]
                else:
                    dataset_split = hf_dataset
                
                # Convert to list of dicts
                self.data = []
                # Auto-detect text column if not specified
                if text_column is None:
                    # Common column names
                    possible_columns = ['text', 'conversation', 'input', 'content', 'prompt', 'messages']
                    text_column = next(
                        (col for col in possible_columns if col in dataset_split.column_names),
                        dataset_split.column_names[0] if dataset_split.column_names else None
                    )
                    if text_column:
                        print(f"Auto-detected text column: '{text_column}'")
                    else:
                        raise ValueError("Could not auto-detect text column. Please specify --text_column")
                
                # Convert dataset to list format
                for item in dataset_split:
                    text = item.get(text_column, "")
                    # Handle different formats
                    if isinstance(text, list):
                        # If it's a list of messages, join them
                        text = self.turn_separator.join(str(msg) for msg in text)
                    elif isinstance(text, dict):
                        # If it's a dict (e.g., with 'user' and 'assistant' keys), format it
                        text = self._format_message_dict(text)
                    self.data.append({"conversation": str(text)})
                
                print(f"Loaded {len(self.data)} examples from HuggingFace dataset")
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace dataset '{data_path}': {e}. "
                               f"Make sure the dataset name is correct and 'datasets' library is installed.")
        else:
            # Local file
            print(f"Loading dataset from local file: {data_path}")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        
        # Ensure data is a list of dicts with 'conversation' key
        if self.data and isinstance(self.data, dict):
            # Convert dict to list if needed
            self.data = [self.data]
        
        # Validate data format
        if not self.data or not isinstance(self.data, list):
            raise ValueError("Data must be a list of dictionaries")
        
        if not all(isinstance(item, dict) for item in self.data):
            raise ValueError("Each item in data must be a dictionary")
        
        # Ensure all items have 'conversation' key
        for i, item in enumerate(self.data):
            if 'conversation' not in item:
                # Try to find alternative keys
                for key in ['text', 'input', 'content', 'prompt']:
                    if key in item:
                        item['conversation'] = item[key]
                        break
                if 'conversation' not in item:
                    raise ValueError(f"Item {i} does not have 'conversation' key. Available keys: {item.keys()}")
    
    def _format_message_dict(self, msg_dict: dict) -> str:
        """
        Format a message dictionary into a conversation string.
        Handles various formats like {'user': ..., 'assistant': ...} or chat format.
        """
        # Try common formats
        if 'user' in msg_dict and 'assistant' in msg_dict:
            # Format with role tags if using im_start format
            if self.turn_separator == "<|im_start|>":
                return f"<|im_start|>user\n{msg_dict['user']}<|im_end|>{self.turn_separator}assistant\n{msg_dict['assistant']}<|im_end|>"
            else:
                return f"{msg_dict['user']}{self.turn_separator}{msg_dict['assistant']}"
        elif 'role' in msg_dict and 'content' in msg_dict:
            # Format with role tag if using im_start format
            if self.turn_separator == "<|im_start|>":
                role = msg_dict.get('role', 'user')
                content = msg_dict.get('content', '')
                return f"<|im_start|>{role}\n{content}<|im_end|>"
            else:
                return msg_dict['content']
        elif 'messages' in msg_dict:
            # Chat format with messages array
            messages = msg_dict['messages']
            parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    if self.turn_separator == "<|im_start|>":
                        role = msg.get('role', 'user')
                        content = msg.get('content', msg.get('text', str(msg)))
                        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                    else:
                        content = msg.get('content', msg.get('text', str(msg)))
                        parts.append(str(content))
                else:
                    parts.append(str(msg))
            return self.turn_separator.join(parts)
        else:
            # Fallback: join all string values
            return self.turn_separator.join(str(v) for v in msg_dict.values() if isinstance(v, (str, int, float)))
        
    def __len__(self):
        return len(self.data)
    
    def find_turn_boundaries(
        self, 
        token_ids: List[int],
        turn_separator_ids: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Find boundaries of turns in tokenized conversation.
        
        Args:
            token_ids: List of token IDs
            turn_separator_ids: Token IDs for turn separator
            
        Returns:
            List of (start, end) tuples for each turn
        """
        boundaries = []
        start = 0
        i = 0
        
        while i < len(token_ids):
            # Check if we found a turn separator
            if i + len(turn_separator_ids) <= len(token_ids):
                if token_ids[i:i+len(turn_separator_ids)] == turn_separator_ids:
                    if start < i:
                        boundaries.append((start, i))
                    start = i + len(turn_separator_ids)
                    i = start
                    continue
            i += 1
        
        # Add last turn
        if start < len(token_ids):
            boundaries.append((start, len(token_ids)))
        
        return boundaries if boundaries else [(0, len(token_ids))]
    
    def __getitem__(self, idx):
        """
        Get a single conversation example.
        
        Expected data format:
        {
            "conversation": "turn1<|turn|>turn2<|turn|>turn3",
            "next_token": "token"  # Optional, for supervised learning
        }
        """
        item = self.data[idx]
        conversation = item["conversation"]
        
        # Tokenize
        turn_separator_ids = self.tokenizer.encode(
            self.turn_separator, 
            add_special_tokens=False
        )
        
        # Tokenize full conversation
        encoded = self.tokenizer(
            conversation,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        token_ids = encoded["input_ids"][0].tolist()
        
        # Find turn boundaries
        turn_boundaries = self.find_turn_boundaries(token_ids, turn_separator_ids)
        
        # Remove padding tokens from boundaries
        padding_id = self.tokenizer.pad_token_id
        if padding_id:
            # Adjust boundaries to exclude padding
            actual_length = len([t for t in token_ids if t != padding_id])
            turn_boundaries = [
                (start, min(end, actual_length)) 
                for start, end in turn_boundaries
                if start < actual_length
            ]
        
        # Create labels for next token prediction
        # Shift input_ids by 1 for next token prediction
        labels = token_ids[1:] + [self.tokenizer.pad_token_id or -100]
        labels = torch.tensor(labels)
        
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": labels,
            "turn_boundaries": turn_boundaries,
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "turn_boundaries": [item["turn_boundaries"] for item in batch],
    }

