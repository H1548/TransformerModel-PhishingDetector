import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from utils import get_batch, estimate_loss
import math
from tokenizers import Tokenizer
from modelpretrain import Transformer


tokenize = Tokenizer.from_file("tokenizer.json")

learning_rate = 6e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_layers = 10
max_iters = 350000
eval_iters = 500
checkpoint_num = 10000
warmup_steps = 32000
hold_steps  = 98000
batch_size = 64
block_size = 110
vocab_size = tokenize.get_vocab_size()
mask_token = tokenize.token_to_id('[MASK]')
mask_spread = 0.15


model = Transformer(num_layers)
m = model.to(device)


optimizer = optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = 0.1)

checkpoint = torch.load('ImprovedPreModelCheckpoint2/checkpoint_epoch_310000.tar')

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    elif current_step < warmup_steps + hold_steps:
        return 1.0
    else:   
        progress = (current_step - warmup_steps - hold_steps) / float(max(1, max_iters - warmup_steps  - hold_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = LambdaLR(optimizer, lr_lambda)


model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_iter = checkpoint['iter'] + 1
best_val = checkpoint.get('val_acc', float('inf'))


losses = estimate_loss(model, eval_iters)
for iter in range(start_iter,max_iters):
    if iter % eval_iters == 0 or iter == max_iters -1:
        losses = estimate_loss(model, eval_iters)
        print(f"step{iter}: train loss {losses['train']:.4f} val loss {losses['val']:.4f}")
    xb,yb = get_batch('train',block_size, batch_size,vocab_size, mask_spread,mask_token, device, tokenize)
    
    logit,loss = model(xb,yb)
    if iter % checkpoint_num == 0 or iter == max_iters-1:
        print(f"Saving Model's Checkpoint for iter:{iter}")
        torch.save({
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': losses['val'],
        }, f'ImprovedPreModelCheckpoint2/checkpoint_epoch_{iter}.tar')

    if losses['val'] < best_val:
        print(f"Saving Best Model...")
        best_val =  losses['val']
        torch.save({
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': losses['val'],
        }, f'ImprovedPreBest2/BestModel.tar')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()