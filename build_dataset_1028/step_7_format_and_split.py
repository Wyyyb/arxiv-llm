import os
import re
from transformers import AutoTokenizer
import json
import time
import torch
import random
from tqdm import tqdm


def batch_compute_tokens(tokenizer, text_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device", device)
    encoded = tokenizer(text_list, add_special_tokens=True, padding=True,
                        truncation=True, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    token_num_list = encoded['input_ids'].ne(tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
    return token_num_list


def split_data():













