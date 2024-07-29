import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from utils import ToSoccerMapTensor

class PFFDataset(Dataset):
    def __init__(self, train_directory, test_directory=None, split_ratio=0.8):
        self.train_data = []
        self.train_labels = []
        self.train_mask = []

        self.val_data = []
        self.val_labels = []
        self.val_mask = []

        self.test_data = []
        self.test_labels = []
        self.test_mask = []

        self._load_data(train_directory, is_train=True)
        if test_directory:
            self._load_data(test_directory, is_train=False)
        else:
            self.train_data, self.val_data, self.train_labels, self.val_labels, self.train_mask, self.val_mask = train_test_split(
                self.train_data, self.train_labels, self.train_mask, test_size=1-split_ratio, random_state=42
            )
        
    def _load_data(self, directory, is_train=True):
        print(f"Temos {len(os.listdir(directory))} amostras na pasta {'treino' if is_train else 'teste'}")
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                df.dropna(subset=['pass_outcome_type'], inplace=True)
                tensor_converter = ToSoccerMapTensor()
                for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processando amostras do csv {filename}"):
                    player_id = int(row["player_id"])
    
                    # Encontrando a coluna que corresponde à condição
                    passerPlayerColumn = [
                        column.replace('original_pId_player_', '')
                        for column in df.columns
                        if 'original_pId_player' in column and player_id in df[column].values
                    ]
                    
                    if passerPlayerColumn:  # Verifica se a lista não está vazia
                        passerPlayerColumn = int(passerPlayerColumn[0])
                        df.loc[idx, 'carrier_velocity'] = np.sqrt(
                            df.loc[idx, f'vx_player_{passerPlayerColumn}']**2 +
                            df.loc[idx, f'vy_player_{passerPlayerColumn}']**2
                        )
                        df.loc[idx, 'vx_carrier'] = df.loc[idx, f'vx_player_{passerPlayerColumn}']
                        df.loc[idx, 'vy_carrier'] = df.loc[idx, f'vy_player_{passerPlayerColumn}']
                        
                    sample = {
                        "ball_x_start": row["ball_x_start"],
                        "ball_y_start": row["ball_y_start"],
                        "ball_x_end": row["ball_x_end"],
                        "ball_y_end": row["ball_y_end"],
                        "pass_outcome_type": row["pass_outcome_type"],
                        "team_id": row["team_id"],
                        "vx_carrier": df.loc[idx, 'vx_carrier'],
                        "vy_carrier": df.loc[idx, 'vx_carrier'],
                        "carrier_velocity": df.loc[idx, 'carrier_velocity'],
                        "frame": df.loc[[idx]],
                    }
                    
                    # Transforma a amostra e obtém a máscara e o target
                    matrix, mask, target = tensor_converter(sample)
                    
                    if is_train:
                        self.train_data.append(matrix)
                        self.train_mask.append(mask)
                        self.train_labels.append(int(target[0]))
                    else:
                        self.test_data.append(matrix)
                        self.test_mask.append(mask)
                        self.test_labels.append(int(target[0]))
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        matrix = self.train_data[idx]
        mask = self.train_mask[idx]
        target = self.train_labels[idx]
        return matrix, mask, target
    
    def get_validation_data(self):
        val_data = torch.stack(self.val_data)
        val_labels = torch.tensor(self.val_labels, dtype=torch.long)
        val_mask = torch.stack(self.val_mask)
        val_dataset = TensorDataset(
            val_data,
            val_mask,
            val_labels
        )
        return val_dataset
    
    def get_test_data(self):
        test_data = torch.stack(self.test_data)
        test_labels = torch.tensor(self.test_labels, dtype=torch.long)
        test_mask = torch.stack(self.test_mask)
        test_dataset = TensorDataset(
            test_data,
            test_mask,
            test_labels
        )
        return test_dataset



if __name__ == "__main__":
    train_directory = '/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/data/Pass'
    teste_directory = '/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/data/Test_Pass'
    dataset = PFFDataset(train_directory,teste_directory, split_ratio=0.8)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset.get_validation_data(), batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset.get_test_data(), batch_size=32, shuffle=False)


    for batch_idx, (data, mask, target) in enumerate(test_loader):
        print(f'Batch {batch_idx + 1}:')
        print('Data:')
        print(data)
        print('Mask:')
        print(mask)
        print('Target:')
        print(target)
        print('---')
goal_x_left, goal_y_left = -52.5, 0
goal_x_right, goal_y_right = 52.5, 0


