import torch
import math
from tokenizers import Tokenizer
from utils import estimate_fineloss, calculate_metrics, create_mask, get_finebatch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from FineTuningmodel import Transformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenize = Tokenizer.from_file("tokenizer.json")

num_layers = 8

model = Transformer(num_layers)
m = model.to(device)

pad_token = tokenize.token_to_id('[PAD]')

batch_size = 54
block_size = 200
learning_rate = 4e-6
checkpoint_num = 10000
warmup_steps = 24000
hold_steps  = 200000
max_iters = 300000
eval_iters = 500

checkpoint = torch.load('MultiFineModelCheckpoint/checkpoint_epoch_260000.tar')
model.load_state_dict(checkpoint['model_state_dict'])

# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and 'fl' not in k}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)

for param in model.parameters():
    param.requires_grad = False
for param in model.fl.parameters():
    param.requires_grad = True


# model._init_weights(model.fl)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=learning_rate, weight_decay=0.1)

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    elif current_step < warmup_steps + hold_steps:
        return 1.0
    else:   
        progress = (current_step - warmup_steps - hold_steps) / float(max(1, max_iters - warmup_steps  - hold_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = LambdaLR(optimizer, lr_lambda)

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


start_iter = checkpoint['iter']+ 1
best_val = checkpoint['val_acc'] 
# best_val = 100

for iter in range(start_iter,max_iters):
    if iter % eval_iters == 0 or iter == max_iters -1:
        losses = estimate_fineloss(model, eval_iters,pad_token)
        metrics = calculate_metrics(model, pad_token, accuracy_score, f1_score, recall_score, precision_score)
        print(f"step{iter}: train loss {losses['train']:.4f} val loss {losses['val']:.4f} accuracy {metrics['accuracy']:.2f} f1_score {metrics['f1']:.3f} recall {metrics['recall']:.3f} precision {metrics['precision']:.3f}")
    xb,yb= get_finebatch('train', block_size, batch_size, pad_token)
    mask = create_mask(xb, pad_token)
    logit,loss = model(xb,mask,yb)
    if iter % checkpoint_num == 0 or iter == max_iters-1:
        print(f"Saving Model's Checkpoint for iter:{iter}")
        torch.save({
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': losses['val'],
        }, f'MultiFineModelCheckpoint/checkpoint_epoch_{iter}.tar')

    if losses['val'] < best_val:
        print(f"Saving Best Model...")
        best_val =  losses['val']
        torch.save({
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': losses['val'],
        }, f'MultiFineBest/BestModel.tar')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    