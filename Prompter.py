import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import random
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tokenizers import Tokenizer
import mmap
from torch.optim.lr_scheduler import LambdaLR
import math
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
from utils import tokenize_data, pad_sequence, create_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_LAUNCH_BLOCKING=1

tokenize = Tokenizer.from_file("tokenizer.json")

vocab_size = tokenize.get_vocab_size()
pad_token = tokenize.token_to_id('[PAD]')
unk_token = tokenize.token_to_id('[UNK]')
mask_token = tokenize.token_to_id('[MASK]')
bos_token  = tokenize.token_to_id("[BOS]")
eos_token = tokenize.token_to_id("[EOS]")
mask_spread = 0.15

batch_size = 54
block_size = 200
num_layers = 8
n_embd = 576
num_heads = 9
hidden = n_embd * 4
dropout = 0.2
learning_rate = 6e-6
max_iters = 150000
eval_iters = 500
checkpoint_num = 10000
warmup_steps = 10000
hold_steps  = 90000
num_classes = 14

label_map = {
    'phishing-login': 0,
    'phishing-financial': 1, 
    'phishing-delivery': 2, 
    'phishing-techsupport': 3, 
    'phishing-tax': 4, 
    'phishing-invoice': 5, 
    'phishing-spear': 6,
    'phishing-malware': 7, 
    'phishing-crypto': 8, 
    'phishing-job': 9, 
    'safe-business': 10, 
    'safe-personal': 11,
    'safe-marketing': 12, 
    'safe-transactional': 13,
    
}

index_to_label = {v:k for k, v in label_map.items()}


class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size,bias=False)
        self.query = nn.Linear(n_embd, head_size,bias=False)
        self.value = nn.Linear(n_embd, head_size,bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
   
        mask = mask.unsqueeze(1)
        wei = (q @ k.transpose(-2,-1)) / self.head_size**0.5
        wei = wei.masked_fill(~mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v 
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,mask):
        out = torch.cat([h(x,mask) for h in self.heads], axis = -1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden) ,
            nn.ReLU(),
            nn.Linear(hidden, n_embd),
            nn.Dropout(dropout)
        ) 
    def forward(self, x):
        out = self.net(x)
        return out
    
class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd, hidden)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x, mask): 
        x = x + self.ln1(self.sa(x,mask))
        x = x + self.ln2(self.ffwd(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb= nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, num_heads=num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.fl = nn.Linear(n_embd, num_classes)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,x,mask,y=None):
        B,T = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x,mask)
        x = self.ln_f(x)
        logits = self.fl(x[:,0,:])
        if y is None:
            loss = None
        else:
            targets = y
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def prompt(self, prompt):
        with torch.no_grad():
            tok_x = [bos_token] + tokenize.encode(prompt).ids + [eos_token]
            padded_x =pad_sequence(tok_x, block_size, pad_token)
            x = torch.tensor(padded_x).unsqueeze(0).to(device)
            mask = create_mask(x, pad_token)
            logits,_ = self(x,mask)
            preds = torch.argmax(logits, dim = 1)
            out = index_to_label[preds.item()]
        return out

AI_DISCLAIMER = "Note: This recommendation was generated by an AI system. Please follow the company's security policy if unsure."

with open("Advice.json","r") as f:
    Response_map = json.load(f)

def generate_response(label): 
    if label in Response_map:
        response_data = Response_map[label]
    else: 
        response_data = {
            "recommended_action" : "Proceed with caution. Unable to classify email type.", 
            "user_advice": "Avoid clicking any links or replying to the sender. Contact your IT team if unsure."
        }
    
    return (f'{label}: Recommended Action: {response_data["recommended_action"]} User Advice: {response_data["user_advice"]}\nAi Disclaimer: {AI_DISCLAIMER}')
# { 
#         "label": label,
#         "recommended_action": response_data["recommended_action"],
#         "user_advice": response_data["user_advice"],
#         "ai_disclaimer": AI_DISCLAIMER
#     }

model = Transformer(num_layers)
m = model.to(device)

checkpoint = torch.load('MultiFineBest/BestModel.tar')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


prompt = input('Enter Your Email: ')
output = model.prompt(prompt)


result = generate_response(output)
print(result)
