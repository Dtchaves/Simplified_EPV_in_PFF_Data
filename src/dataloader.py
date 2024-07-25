import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


from utils import ToSoccerMapTensor

class PFFDataset(Dataset):
    def __init__(self, directory, split_ratio=0.8):
        self.data = []
        self.labels = []
        self.mask = []  # Adicione a lista de máscara
        self._load_data(directory)
        self.train_data, self.test_data, self.train_labels, self.test_labels, self.train_mask, self.test_mask = train_test_split(
            self.data, self.labels, self.mask, test_size=1-split_ratio, random_state=42
        )
        
    def _load_data(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                df.dropna(subset=['pass_outcome_type'], inplace=True)
                tensor_converter = ToSoccerMapTensor()
                
                for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processando amostras do csv {filename}"):
                    sample = {
                        "ball_x_start": row["ball_x_start"],
                        "ball_y_start": row["ball_y_start"],
                        "ball_x_end": row["ball_x_end"],
                        "ball_y_end": row["ball_y_end"],
                        "vx_player_201": row["vx_player_201"],
                        "vy_player_201": row["vy_player_201"],
                        "pass_outcome_type": row["pass_outcome_type"],
                        "team_id": row["team_id"],
                        "carrier_velocity": row["carrier_velocity"],
                        "frame": df[df["team_id"] == row["team_id"]]
                    }
                    
                    # Transforme a amostra e obtenha a máscara e o target
                    matrix, mask, target = tensor_converter(sample)
                    
                    self.data.append(matrix)
                    self.mask.append(mask)
                    self.labels.append(int(target[0])) 
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        matrix = self.train_data[idx]
        mask = self.train_mask[idx]
        target = self.train_labels[idx]
        return matrix, mask, target
    
    def get_validation_data(self):
        test_data = torch.stack(self.test_data)
        test_labels = torch.tensor(self.test_labels, dtype=torch.long)
        test_mask = torch.stack(self.test_mask)
        test_dataset = torch.utils.data.TensorDataset(
            test_data,
            test_mask,
            test_labels
        )
        return test_dataset
    

if __name__ == "__main__":
    directory = '/home/diogo/Documents/Simplified_EPV_in_PFF_Data/data'
    dataset = PFFDataset(directory, split_ratio=0.8)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset.get_validation_data(), batch_size=32, shuffle=False)

    for batch_idx, (data, mask, target) in enumerate(train_loader):
        print(f'Batch {batch_idx + 1}:')
        print('Data:')
        print(data)
        print('Mask:')
        print(mask)
        print('Target:')
        print(target)
        print('---')
