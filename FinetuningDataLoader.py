import pandas as pd
import random 
from utils import tokenize_data

def load_finedata(data_path):
    df = pd.read_csv(data_path, index_col = None,skip_blank_lines = True)
    df.columns = ['email', 'label']
    df = df.dropna(subset=['label']).reset_index(drop=True)

    label_map = {
        'phishing-login': 0,
        'phishing-financial': 1, 
        'phishing-delivery': 2, 
        'phishing-techsupport': 3, 
        'phishing-tax': 4, 
        'phishing-invoice': 5, 
        'phishing-spear': 6,
        'phishing-job': 7, 
        'safe-business': 8, 
        'safe-personal': 9,
        'safe-marketing': 10, 
        'safe-transactional': 11,
        
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