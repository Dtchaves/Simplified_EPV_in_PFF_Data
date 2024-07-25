import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CSVDataset(Dataset):
    def __init__(self, directory, split_ratio=0.8):
        self.data = []
        self.labels = []
        self._load_data(directory)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.data, self.labels, test_size=1-split_ratio, random_state=42
        )
        
    def _load_data(self, directory):
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                for _, row in df.iterrows():
                    self.data.append(row[:-1].values)  
                    self.labels.append(row[-1])
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.train_data[idx], dtype=torch.float32)
        label = torch.tensor(self.train_labels[idx], dtype=torch.long)
        return sample, label
    
    def get_test_data(self):
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(self.test_data, dtype=torch.float32),
            torch.tensor(self.test_labels, dtype=torch.long)
        )
        return test_dataset

# Use the CSVDataset
directory = 'path/to/your/csv/files'
dataset = CSVDataset(directory, split_ratio=0.8)

# Create DataLoaders
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset.get_test_data(), batch_size=32, shuffle=False)
