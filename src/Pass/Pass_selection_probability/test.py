import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import sklearn.metrics as metrics
import pandas as pd
import random
from sklearn.preprocessing import QuantileTransformer

from dataclasses import dataclass, field
from soccermap import SoccerMapPassSelect, pixel
from torch.utils.data import DataLoader
from dataloader import PFFDataset




class TestPassSelect:
    def __init__(self, model_path, train_directory, test_directory, dataset, model_name, path_metric, path_heatmap, device):
        self.model_path = model_path
        self.model = torch.load(self.model_path)
        self.train_directory = train_directory
        self.test_directory = test_directory
        self.dataset = dataset
        self.test_loader = DataLoader(self.dataset.get_test_data(), batch_size=32, shuffle=False)
        self.model_name = model_name
        self.path_metric = path_metric
        self.path_heatmap = path_heatmap
        self.device = device

    def metric(self, y_true, y_pred):
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, average='macro')
        recall = metrics.recall_score(y_true, y_pred, average='macro')
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        report_table = {
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1]
        }

        report_table_df = pd.DataFrame(report_table)

        fig, ax = plt.subplots(figsize=(20, 1))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=report_table_df.values, colLabels=report_table_df.columns, cellLoc='center', loc='center')
        table.scale(1, 2)

        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(weight='bold')
        name = f"Metrics_" + self.model_name
        save_path = os.path.join(self.path_metric, self.model_name)
        plt.savefig(save_path)

    def Get_Y(self):
        y_true = []
        y_pred = []
        i = 0
        with torch.no_grad():
            for matriz, mask, target in self.test_loader:
                matriz = matriz.to(self.device)
                mask = mask.to(self.device).float()
                label = torch.ones(32, device=self.device).float()

                surface = self.model(matriz)

                pred = pixel(surface, mask).view(-1)
                cutoff = 0.75
                pred = (pred >= cutoff).float()

                y_true.append(label)
                y_pred.append(pred)

        y_true = torch.cat(y_true).to('cpu')
        y_pred = torch.cat(y_pred).to('cpu')
        print(type(y_true))
        print(type(y_pred))
        return y_true, y_pred

    def plot_random_heatmap(self):
        # Selecionar um índice aleatório de um lote
        random_index = random.randint(0, len(self.test_loader.dataset) - 1)

        with torch.no_grad():
            matriz, mask, target = self.test_loader.dataset[4562]

            plt.figure(figsize=(10, 8), dpi=100)  # Ajustar o tamanho e DPI da figura
            plt.imshow(matriz[2], cmap='jet', interpolation='gaussian', aspect='auto')
            plt.colorbar()
            plt.title("Heatmap Aleatório da saída do SoccerMap da bola")
            plt.xlabel("Largura")
            plt.ylabel("Altura")
            save_path = os.path.join(self.path_heatmap, f"{self.model_name}_bola.png")
            plt.savefig(save_path)

            plt.figure(figsize=(10, 8), dpi=100)  # Ajustar o tamanho e DPI da figura
            plt.imshow(matriz[0], cmap='jet', interpolation='gaussian', aspect='auto')
            plt.colorbar()
            plt.title("Heatmap Aleatório da saída do SoccerMap dos jogadores ataque")
            plt.xlabel("Largura")
            plt.ylabel("Altura")
            save_path = os.path.join(self.path_heatmap, f"{self.model_name}_jogadoresa.png")
            plt.savefig(save_path)

            plt.figure(figsize=(10, 8), dpi=100)  # Ajustar o tamanho e DPI da figura
            plt.imshow(matriz[1], cmap='jet', interpolation='gaussian', aspect='auto')
            plt.colorbar()
            plt.title("Heatmap Aleatório da saída do SoccerMap dos jogadores defesa")
            plt.xlabel("Largura")
            plt.ylabel("Altura")
            save_path = os.path.join(self.path_heatmap, f"{self.model_name}_jogadoresd.png")
            plt.savefig(save_path)
            
            matriz = matriz.unsqueeze(0).to(self.device)
            surface = self.model(matriz)

            output_np = surface[0].cpu().detach().numpy().squeeze()
            scaler = QuantileTransformer(output_distribution='uniform')
            #output_np = scaler.fit_transform(output_np)
            
            plt.figure(figsize=(10, 8), dpi=100)  # Ajustar o tamanho e DPI da figura
            plt.imshow(output_np, cmap='jet', interpolation='gaussian', aspect='auto')
            plt.colorbar()
            plt.title("Heatmap Aleatório da saída do SoccerMap")
            plt.xlabel("Largura")
            plt.ylabel("Altura")
            save_path = os.path.join(self.path_heatmap, f"{self.model_name}_random.png")
            plt.savefig(save_path)

    def run(self):
        self.model.eval()
        y_true, y_pred = self.Get_Y()
        self.metric(y_true=y_true, y_pred=y_pred)
        classification_report = metrics.classification_report(y_true, y_pred, target_names=None)
        self.plot_random_heatmap()
        return classification_report

@dataclass
class TestConfig:
    model_path: str = "/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/results/models/Pass_selection_probability.pt"
    train_directory: str = '/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/data/Vazia'
    test_directory: str = '/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/data/Test_Pass'
    dataset: PFFDataset = PFFDataset(train_directory, test_directory, split_ratio=0.8)
    model_name: str = "Pass_selection_probability"
    path_metric: str = "/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/results/metrics"
    path_heatmap: str = "/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/results/heatmaps"
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Test():
    config = TestConfig()
    test = TestPassSelect(**config.__dict__)
    test.run()

if __name__ == "__main__":
    Test()
