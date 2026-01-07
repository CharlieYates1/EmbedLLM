import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from transformers import AutoModelForCausalLM, AutoTokenizer
llm_name: str = "Qwen/Qwen3-0.6B"
llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    trust_remote_code=True,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(llm_name)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1

# llm = get_peft_model(llm, LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=lora_r,
#     lora_alpha=lora_alpha,
#     lora_dropout=lora_dropout,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=False,
# ))
llm = PeftModel.from_pretrained(llm, "checkpoints/base_llm_qwen3-0.6b_lora/with_embed", is_trainable=True)
llm.print_trainable_parameters()

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

num_vectors = 32

class EmbeddingToQwenCrossAttentionModel(nn.Module):
    """
    Model that takes input text, encodes it with an embedding model,
    and projects it to 32 vectors of dimension equal to Qwen's hidden_dim
    using a cross-attention mechanism.
    """
    def __init__(self, 
                 embedding_model, 
                 embedding_dim: int,
                 qwen_hidden_dim: int, 
                 num_vectors: int = 32,
                 embed_tokenizer=None):
        super().__init__()
        self.embedding_model = embedding_model  # must take text and return (batch, embed_dim)
        self.embed_tokenizer = embed_tokenizer  # Tokenizer for embedding model
        self.qwen_hidden_dim = qwen_hidden_dim
        self.num_vectors = num_vectors

        # Linear to project embedding model's output to a "memory" for cross-attention
        self.memory_proj = nn.Linear(
            embedding_dim, qwen_hidden_dim
        )

        # Learnable queries to use for cross-attention
        self.queries = nn.Parameter(torch.randn(1, num_vectors, qwen_hidden_dim))

        # Cross-attention module: query = [num_vectors, hidden_dim], key & value = [seq, hidden_dim]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=qwen_hidden_dim,
            num_heads=4,
            batch_first=True
        )

    def freeze_embedding_model(self):
        for param in self.embedding_model.parameters():
            param.requires_grad = False

    def forward(self, text_input):
        """
        text_input: Can be either:
            - List of strings (raw text)
            - Tensor of token IDs (must be from embedding model's tokenizer)
        Returns: projected_vectors: shape (batch, num_vectors, qwen_hidden_dim)
        """
        # 1. Encode text to embedding
        # If text_input is a tensor, assume it's already tokenized for embedding model
        # If it's strings or needs tokenization, tokenize with embedding model's tokenizer
        if isinstance(text_input, list) and isinstance(text_input[0], str):
            # Raw text - tokenize with embedding model's tokenizer
            if self.embed_tokenizer is None:
                raise ValueError("embed_tokenizer required when passing raw text")
            encoded = self.embed_tokenizer(
                text_input,
                padding=True,
                truncation=True,
                max_length=512,  # E5 models typically use 512 max length
                return_tensors="pt"
            )
            # Move to same device as embedding model
            device = next(self.embedding_model.parameters()).device
            embedding_input_ids = encoded["input_ids"].to(device)
            embedding = self.embedding_model(input_ids=embedding_input_ids)
        elif isinstance(text_input, torch.Tensor):
            # Assume it's already tokenized for embedding model
            embedding = self.embedding_model(input_ids=text_input)
        else:
            raise ValueError(f"Unsupported input type: {type(text_input)}")
        if hasattr(embedding, "last_hidden_state"):  # For HF models
            if len(embedding.last_hidden_state.shape) == 3:
                # Use [CLS] or mean pooling
                emb_vec = embedding.last_hidden_state.mean(dim=1)
            else:
                emb_vec = embedding.last_hidden_state
        elif isinstance(embedding, torch.Tensor):
            # (batch, embed_dim)
            emb_vec = embedding
        else:
            raise ValueError("Unknown embedding_model output structure")

        # 2. Project embedding to Qwen hidden dimension
        memory = self.memory_proj(emb_vec).unsqueeze(1)  # (batch, 1, qwen_hidden_dim)

        # 3. Prepare queries: expand learnable [1, num_vectors, h] to batch
        queries = self.queries.expand(emb_vec.shape[0], -1, -1)  # (batch, num_vectors, dim)

        # 4. Cross attention: query=(B, N, D), key/value=(B, 1, D)
        # nn.MultiheadAttention expects shape (batch_size, seq_length, embed_dim)
        attended, _ = self.cross_attn(queries, memory, memory)  # (batch, num_vectors, dim)
        return attended

    def load_projector(self, path):
        import os

        additional_path = os.path.join(path, "additional_components.pt")
        additional_components = torch.load(additional_path, map_location=device)
        self.memory_proj.load_state_dict(additional_components["memory_proj"])
        # queries is a Parameter, load it directly
        if "queries" in additional_components:
            self.queries.data.copy_(additional_components["queries"])
        self.cross_attn.load_state_dict(additional_components["cross_attn"])

embedding_model = "Qwen/Qwen3-Embedding-0.6B"
embed_model = AutoModel.from_pretrained(embedding_model)
# embed_model = get_peft_model(embed_model, LoraConfig(
#     task_type=TaskType.FEATURE_EXTRACTION,  # BERT models are encoders for feature extraction
#     r=lora_r,
#     lora_alpha=lora_alpha,
#     lora_dropout=lora_dropout,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     inference_mode=False,
# ))
embed_model = PeftModel.from_pretrained(embed_model, "checkpoints/base_llm_qwen3-0.6b_lora/with_qwen_embed/embed_model", is_trainable=True)
embed_model.print_trainable_parameters()
embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model)  # Need embedding model's tokenizer
cross_proj = EmbeddingToQwenCrossAttentionModel(
    embedding_model=embed_model,
    embedding_dim=embed_model.config.hidden_size,
    qwen_hidden_dim=llm.config.hidden_size,
    num_vectors=num_vectors,
    embed_tokenizer=embed_tokenizer,  # Pass embedding model's tokenizer
).to(device)
# cross_proj.freeze_embedding_model()
cross_proj.load_projector("checkpoints/base_llm_qwen3-0.6b_lora/with_qwen_embed")
class QwenWithCrossAttention(nn.Module):
    def __init__(self, qwen_model, cross_proj):
        super().__init__()
        self.qwen_model = qwen_model
        self.cross_proj = cross_proj

    def forward(self, input_ids, attention_mask, labels, text=None):
        # Encode text with embedding model (needs raw text, not Qwen token IDs)
        if text is None:
            raise ValueError("text parameter required for embedding model")
        embeddings = self.cross_proj(text)  # Pass raw text, not input_ids
        embed_layer = self.qwen_model.get_input_embeddings()
        text_embeddings = embed_layer(input_ids)
        total_embeddings = torch.cat([embeddings, text_embeddings], dim=1)
        
        # Pass embeddings to Qwen model
        outputs = self.qwen_model(inputs_embeds=total_embeddings, attention_mask=attention_mask, labels=labels)
        
        return outputs

    def get_num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

model = QwenWithCrossAttention(llm, cross_proj)
print(model.get_num_trainable_parameters())

from datasets import load_dataset
from data_utils import get_clean_turns
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Optional
import torch

data_path = "Bossologist/general_Qwen3_ft_dataset"

class ConversationDataset(Dataset):
    """
    Dataset for conversation data with turn boundaries.
    Supports both local JSON files and HuggingFace datasets.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
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
        
        print(f"Loading dataset from HuggingFace: {data_path}")
        hf_dataset = load_dataset(data_path)
        dataset_split = hf_dataset['train']
        
        # Convert to list of dicts
        self.data = []
        # Auto-detect text column if not specified
        possible_columns = ['text', 'conversation', 'input', 'content', 'prompt', 'messages']
        text_column = next(
            (col for col in possible_columns if col in dataset_split.column_names),
            dataset_split.column_names[0] if dataset_split.column_names else None
        )
        for item in dataset_split:
            text = item.get(text_column)    
            turns = get_clean_turns(text)
            for i in range(len(turns)):
                if turns[i]["role"] != "system":
                    self.data.append({"conversation": turns[i]['content'] + "<|im_end|>"})
        
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
        
        # Tokenize full conversation
        encoded = self.tokenizer(
            conversation,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        token_ids = encoded["input_ids"][0].tolist()
        attention_mask = encoded["attention_mask"][0].tolist()
        
        # Create labels for next token prediction
        # Shift input_ids by 1 for next token prediction
        labels = token_ids
        labels = torch.tensor(labels).to(dtype=torch.long)
        labels[torch.tensor(attention_mask) == 0] = -100
        labels = labels[:-num_vectors]
        # Pad each label tensor with number of vectors (-100) at the start
        pad = torch.full((num_vectors,), -100, dtype=labels.dtype)
        labels = torch.cat([pad, labels])

        attention_mask = encoded["attention_mask"][0]
        pad = torch.full((num_vectors,), 1, dtype=labels.dtype)
        attention_mask = torch.cat([pad, attention_mask])
        attention_mask = attention_mask[:-num_vectors]
        return {
            "input_ids": encoded["input_ids"][0][:-num_vectors],
            "attention_mask": attention_mask,
            "labels": labels,
            "text": conversation,  # Store raw text for embedding model
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "text": [item["text"] for item in batch],  # Keep as list of strings
    }

dataset = ConversationDataset(
    data_path=data_path,
    tokenizer=tokenizer,
)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn,
)

from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Optimizer tracking {len(trainable_params)} parameter groups (trainable only)")
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=5e-5,
    weight_decay=0.01,
)

# Learning rate scheduler
num_epochs = 1
num_training_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
accelerator = Accelerator()
model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)

from tqdm import tqdm

import os

def save_model(model, output_path, cur_dir):
    print("Saving final model...")
    save_dir = os.path.join(output_path, cur_dir)
    os.makedirs(save_dir, exist_ok=True)

    additional_components = {}
    additional_components["memory_proj"] = model.cross_proj.memory_proj.state_dict()
    # queries is a Parameter, not a module, so just save it directly
    additional_components["queries"] = model.cross_proj.queries.data.clone()
    additional_components["cross_attn"] = model.cross_proj.cross_attn.state_dict()
    torch.save(additional_components, os.path.join(save_dir, f"additional_components.pt"))
    model.qwen_model.save_pretrained(save_dir)
    
    embed_model_path = os.path.join(save_dir, "embed_model")
    os.makedirs(embed_model_path, exist_ok=True)
    embed_model.save_pretrained(embed_model_path)

global_step = 0
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            text=batch["text"],  # Pass raw text for embedding model
        )
        loss = output.loss
        
        # Backward pass
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        global_step += 1
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item(), "avg_loss": epoch_loss / global_step})
        if global_step % 10000 == 0:
            save_model(model, "checkpoints/base_llm_qwen3-0.6b_lora", f"with_qwen_embed-{global_step}")

        if global_step % 100 == 0:
            print(f"Global step {global_step}")
            print(f"Input: {batch['text'][0]}")
            print(f"Output: {tokenizer.decode(output.logits.argmax(dim=-1)[0][num_vectors - 1:], skip_special_tokens=False)}")

    save_model(model, "checkpoints/base_llm_qwen3-0.6b_lora", f"with_qwen_embed")
    
    # Epoch summary
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")