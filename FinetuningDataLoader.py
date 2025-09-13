import pandas as pd
import random 
from utils import tokenize_data

def load_finedata(data_path):
    df = pd.read_csv(data_path, index_col = None,skip_blank_lines = True)
    df.columns = ['email', 'label']
    df = df.dropna(subset=['label']).reset_index(drop=True)

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

    numerical_labels = [label_map[label] for label in df['label']]
    new_data = tokenize_data(df['email'])

    paired_data = list(zip(new_data,numerical_labels))
    copy_data = paired_data.copy()
    random.shuffle(copy_data)
    n = int(0.8 * len(paired_data))
    train_data = copy_data[:n]
    val_data = copy_data[n:189823]
    test_data = copy_data[189823:189923]

    return train_data, val_data, test_data