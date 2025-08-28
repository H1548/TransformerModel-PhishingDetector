import mmap
import random 
import torch
import torch.nn as nn
from FinetuningDataLoader import load_finedata

device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_path = 'Dataset/MultiLabelPhishing2'

def get_random_chunk(split, block_size, batch_size,tokenize):
    filename = 'train_split.txt' if split == 'train' else 'val_split.txt'
    with open(filename,'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size  = len(mm)
            start_pos = random.randint(0,(file_size) - block_size*batch_size-1)

            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            decoded_block = block.decode('utf-8',errors='ignore').replace('\r','')
            
            data = torch.tensor(tokenize.encode(decoded_block).ids)
    return data

def get_batch(split, block_size, batch_size,vocab_size, mask_spread,mask_token, device, tokenize):
    data  = get_random_chunk(split, block_size, batch_size,tokenize)
    ix = torch.randint(len(data)-block_size, size=(batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.full((batch_size, block_size),-100,dtype = torch.long) 
    x_copy = x.clone()
    
    for b in range(x.shape[0]):
        current_row = x[b]
        num_to_mask  = int(mask_spread * len(current_row))
        idx = torch.randint(len(current_row), size=(num_to_mask,))

        for i in idx:
            prob = torch.rand(1).item()

            if prob < 0.8:
                x_copy[b,i] = mask_token
            elif prob < 0.9:
                random_token = torch.randint(low = 0 , high=vocab_size, size=(1,)).item()
                x_copy[b, i] = random_token
            else: 
                pass
     
            y[b,i] = x[b,i]  
    x_copy,y = x_copy.to(device),y.to(device)
    return x_copy,y

@torch.no_grad()
def estimate_loss(model, eval_iters,split,block_size, batch_size,vocab_size, mask_spread,mask_token, device, tokenize):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split,block_size, batch_size,vocab_size, mask_spread,mask_token, device, tokenize)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def tokenize_data(data, bos_token, eos_token, tokenize):
    for i in range(data.shape[0]):
        data[i] = [bos_token] + tokenize.encode(data[i]).ids + [eos_token]
    return data

def pad_sequence(sequence, max_length, pad_token_id):
    if len(sequence) > max_length:
        return sequence[:max_length]
    return sequence + [pad_token_id] * (max_length - len(sequence))

def create_mask(x, pad_token):
    mask = x != pad_token
    mask = mask.to(torch.bool)
    return mask

def get_finebatch(split, block_size, batch_size, pad_token):
    train_data,val_data, test_data = load_finedata(file_path)
    data = train_data if split == 'train' else val_data
    ix =torch.randint(len(data)-block_size, size=(batch_size,))

    x, y = [], []

    for i in ix:
        input_i, output_i = data[i]
        pad_input_i = pad_sequence(input_i, block_size,pad_token)
        x.append(torch.tensor(pad_input_i, dtype=torch.long))
        y.append(torch.tensor(output_i, dtype = torch.long))
        

    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)

    return x, y

@torch.no_grad()
def estimate_fineloss(model, eval_iters, pad_token):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            mask = create_mask(X, pad_token)
            logits, loss = model(X, mask, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def calculate_metrics(model, pad_token, accuracy_score, f1_score, recall_score, precision_score):
    out = {}
    all_preds = []
    all_targets = []
    model.eval()
    x, y= get_finebatch('val', )
    mask = create_mask(x, pad_token)
    logits,_ = model(x, mask)
    preds = torch.argmax(logits, dim = 1)
    # all_preds.append(preds.cpu().numpy())
    # all_targets.append(y.cpu().numpy())
    preds= preds.cpu().numpy()
    y = y.cpu().numpy()
    accuracy = accuracy_score(y, preds) * 100
    f1 = f1_score(y, preds, average='macro')
    recall = recall_score(y, preds, average='macro')
    precision = precision_score(y, preds, average='macro', zero_division=0)
    out['accuracy'], out['f1'], out['recall'], out['precision'] = accuracy, f1, recall, precision
    model.train()
    return out