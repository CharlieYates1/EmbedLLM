"""
Data utilities for processing conversation data.
"""
import json
import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer

from datasets import load_dataset

def get_clean_turns(conversation: str) -> List[dict]:
    """
    Remove all <think></think> blocks from a conversation string.
    """
    # Convert dataset to list format
    ret = []
    turns = ["<|im_start|>" + t for t in conversation.split("<|im_start|>") if t.strip()]
    for turn in turns:
        role = None
        actual_text = turn.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        if actual_text.startswith("user"):
            actual_text = "".join(actual_text.split("\n")[1:]).strip()
            role = "user"
        elif actual_text.startswith("assistant"):
            actual_text = "".join(actual_text.split("\n")[1:]).strip()
            role = "assistant"
        elif actual_text.startswith("system"):
            actual_text = "".join(actual_text.split("\n")[1:]).strip()
            role = "system"
        else:
            print(f"Unknown role: {actual_text}")
        ret.append({"role": role, "content": actual_text})
    return ret

def get_formatted_conversation(turns: List[dict]) -> str:
    """
    Format a list of turns into a conversation string.
    """
    ret = ""
    for turn in turns:
        ret += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
    return ret


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
        
        print(f"Loading dataset from HuggingFace: {data_path}")
        hf_dataset = load_dataset(data_path)
        dataset_split = hf_dataset['train']
        
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
        for item in dataset_split:
            text = item.get(text_column)    
            turns = get_clean_turns(text)
            for i in range(len(turns) - 1):
                conversation = get_formatted_conversation(turns[:i+1])
                self.data.append({"conversation": conversation, "predictions": get_formatted_conversation([turns[i+1]])})
        
        print(f"Loaded {len(self.data)} examples from HuggingFace dataset")
        
    def __len__(self):
        return len(self.data)
    
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
        predictions = item["predictions"]
        
        # Tokenize full conversation
        encoded = self.tokenizer(
            conversation,
            truncation=True,
            return_tensors="pt",
        )

        encoded_predictions = self.tokenizer(
            predictions,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        token_ids = encoded_predictions["input_ids"][0].tolist()
        attention_mask = encoded_predictions["attention_mask"][0].tolist()
        
        # Create labels for next token prediction
        # Shift input_ids by 1 for next token prediction
        labels = token_ids
        labels = torch.tensor(labels).to(dtype=torch.long)
        labels[torch.tensor(attention_mask) == 0] = -100
        
        return {
            "input_ids": encoded_predictions["input_ids"][0],
            "attention_mask": encoded_predictions["attention_mask"][0],
            "labels": labels,
            "conversation_ids": encoded["input_ids"][0],
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "conversation_ids": torch.stack([item["conversation_ids"] for item in batch]),
    }

