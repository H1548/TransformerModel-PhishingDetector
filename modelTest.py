from test_dataloader import load_testdata
from FineTuningmodel import Transformer
from utils import get_testbatch, estimate_fineloss, calculate_metrics, create_mask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tokenizers import Tokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_LAUNCH_BLOCKING=1

tokenize = Tokenizer.from_file("tokenizer.json")

vocab_size = tokenize.get_vocab_size()
pad_token = tokenize.token_to_id('[PAD]')
unk_token = tokenize.token_to_id('[UNK]')
mask_token = tokenize.token_to_id('[MASK]')
bos_token  = tokenize.token_to_id("[BOS]")
eos_token = tokenize.token_to_id("[EOS]")


batch_size = 50
block_size = 110
num_layers = 10







label_map = {
    'phishing-credential': 0,   
    'phishing-payment': 1,      
    'phishing-delivery': 2,
    'phishing-techsupport': 3,
    'phishing-job': 4,
    'safe-work': 5,             
    'safe-personal': 6,
    'safe-marketing': 7,
    'safe-transactional': 8,
    }


test_data = load_testdata('Dataset/Testset.csv')


model = Transformer(num_layers)
m = model.to(device)
checkpoint = torch.load('MultiFineBest2/BestModel.tar')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

id2label = {v: k for k,v in label_map.items()}
labels = sorted(id2label.keys())
target_names  = [id2label[i] for i in labels]


all_preds, all_labels = [], []
loss_sum = 0.0
n = 0.0
with torch.no_grad():
    for start in range(0, len(test_data), batch_size):
        idx = list(range(start, min(start + batch_size, len(test_data))))
        x, y = get_testbatch(test_data, idx)
        mask = create_mask(x,pad_token)
        logits, loss = model.forward(x,mask,y)
        loss_sum += loss.item() * y.size(0)
        n += y.size(0)

        preds = logits.argmax(dim=-1).detach().cpu()
        all_preds.append(preds)
        all_labels.append(y.detach().cpu())

test_loss = loss_sum / n
y_pred = torch.cat(all_preds).numpy()
y_true = torch.cat(all_labels).numpy()


acc = accuracy_score(y_true, y_pred)
f1w = f1_score(y_true, y_pred, average="weighted")
print(f"Test loss: {test_loss:.4f} | Acc: {acc:.4f} | F1(w): {f1w:.4f}")

report = classification_report(y_true, y_pred,labels=labels,target_names=target_names, digits=4,zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels= labels)

cm_df = pd.DataFrame(
    cm,
    index=[f"true_{c}" for c in target_names],
    columns=[f"pred_{c}" for c in target_names]
)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, ha="right")
plt.yticks(tick_marks, target_names)

# Add numbers inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("Results/confusion_matrix.png")
cm_df.to_csv("Results/confusion_matrix.csv")

report_dict = classification_report(y_true, y_pred, output_dict=True, digits=4)
df = pd.DataFrame(report_dict).transpose()
df.to_csv("Results/classification_report.csv", index=True)




