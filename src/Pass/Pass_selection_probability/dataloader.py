import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from utils import ToSoccerMapTensor

PASS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(PASS_ROOT) not in sys.path:
    sys.path.append(str(PASS_ROOT))

from reward_labels import PassRewardLabeler

class PFFDataset(Dataset):
    def __init__(
        self,
        train_directory,
        test_directory=None,
        split_ratio=0.8,
        label_mode="pass_outcome",
        reward_event_directory="data/raw/event",
        reward_horizon_seconds=15.0,
        include_open_play_null=True,
    ):
        self.train_data = []
        self.train_labels = []
        self.train_mask = []

        self.val_data = []
        self.val_labels = []
        self.val_mask = []

        self.test_data = []
        self.test_labels = []
        self.test_mask = []

        self.label_mode = label_mode
        if self.label_mode not in {"pass_outcome", "reward"}:
            raise ValueError("label_mode must be one of: 'pass_outcome', 'reward'.")

        self.reward_labeler = None
        if self.label_mode == "reward":
            event_root = Path(reward_event_directory)
            if not event_root.is_absolute():
                event_root = (REPO_ROOT / event_root).resolve()
            self.reward_labeler = PassRewardLabeler(
                event_root=event_root,
                horizon_seconds=reward_horizon_seconds,
                include_open_play_null=include_open_play_null,
            )

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

                if self.label_mode == "reward" and self.reward_labeler is not None:
                    df, label_summary = self.reward_labeler.label_pass_dataframe(
                        df,
                        source_filename=filename,
                        drop_unlabeled=True,
                    )
                    if df.empty:
                        continue

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
                        "vy_carrier": df.loc[idx, 'vy_carrier'],
                        "carrier_velocity": df.loc[idx, 'carrier_velocity'],
                        "frame": df.loc[[idx]],
                    }
                    
                    # Transforma a amostra e obtém a máscara e o target
                    matrix, mask, target = tensor_converter(sample)
                    target_value = int(row["reward_label"]) if self.label_mode == "reward" else int(target[0])
                    
                    if is_train:
                        self.train_data.append(matrix)
                        self.train_mask.append(mask)
                        self.train_labels.append(target_value)
                    else:
                        self.test_data.append(matrix)
                        self.test_mask.append(mask)
                        self.test_labels.append(target_value)
        
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
    train_directory = 'passes'
    # teste_directory = '/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/data/Test_Pass'
    dataset = PFFDataset(train_directory,test_directory=None, split_ratio=0.8)

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


