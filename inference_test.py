import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import os

from model import PerceiverQwenModel
from data_utils import ConversationDataset, collate_fn

qwen_model_name = "Bossologist/Qwen3-4B-Instruct-2507_general_ft_merged"
perceiver_model_name = "deepmind/multimodal-perceiver"

tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PerceiverQwenModel(
    qwen_model_name=qwen_model_name,
    perceiver_model_name=perceiver_model_name,
).to("cuda")

