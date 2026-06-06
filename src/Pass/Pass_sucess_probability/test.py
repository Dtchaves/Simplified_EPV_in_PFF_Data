import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
import pandas as pd
import random
import json
from sklearn.preprocessing import QuantileTransformer


from dataclasses import dataclass, field
from torch.utils.data import DataLoader

try:
    from soccermap import SoccerMapPassSucess, pixel
    from dataloader import PFFDataset
except ImportError:
    from .soccermap import SoccerMapPassSucess, pixel
    from .dataloader import PFFDataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _resolve_repo_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)




class TestPassSucess:
    def __init__(self, model_path, train_directory, test_directory, dataset, model_name, path_metric, path_heatmap, device):
        self.model_path = _resolve_repo_path(model_path)
        self.model = torch.load(self.model_path, weights_only=False)
        self.train_directory = train_directory
        self.test_directory = test_directory
        self.dataset = dataset
        self.val_loader = DataLoader(self.dataset.get_validation_data(), batch_size=32, shuffle=False)
        self.test_loader = DataLoader(self.dataset.get_test_data(), batch_size=32, shuffle=False)
        self.model_name = model_name
        self.path_metric = _resolve_repo_path(path_metric)
        self.path_heatmap = _resolve_repo_path(path_heatmap)
        self.device = device
        self.ece_bins = 10

    @staticmethod
    def _prob_to_logit(probabilities: torch.Tensor) -> torch.Tensor:
        clipped = probabilities.clamp(1e-6, 1 - 1e-6)
        return torch.log(clipped) - torch.log1p(-clipped)

    def _collect_binary_outputs(self, loader):
        probabilities = []
        labels = []
        with torch.no_grad():
            for matriz, mask, target in loader:
                matriz = matriz.to(self.device)
                mask = mask.to(self.device).float()
                label = target.to(self.device).float().view(-1)

                surface = self.model(matriz)
                prob = pixel(surface, mask).view(-1).clamp(1e-6, 1 - 1e-6)

                probabilities.append(prob)
                labels.append(label)

        return torch.cat(probabilities), torch.cat(labels)

    def _fit_temperature(self, val_probs: torch.Tensor, val_labels: torch.Tensor) -> float:
        logits = self._prob_to_logit(val_probs).detach()
        labels = val_labels.detach()

        log_temperature = torch.nn.Parameter(torch.zeros(1, device=logits.device))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature).clamp_min(1e-3)
            calibrated_logits = logits / temperature
            loss = F.binary_cross_entropy_with_logits(calibrated_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(torch.exp(log_temperature).detach().item())

    @staticmethod
    def _compute_quantile_ece(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
        probs = probabilities.detach().cpu().float().view(-1)
        truths = labels.detach().cpu().float().view(-1)
        n = probs.numel()
        if n == 0:
            return float("nan")

        quantiles = torch.quantile(probs, torch.linspace(0.0, 1.0, bins + 1))
        ece = torch.tensor(0.0)
        for idx in range(bins):
            lower = quantiles[idx]
            upper = quantiles[idx + 1]
            if idx == bins - 1:
                in_bin = (probs >= lower) & (probs <= upper)
            else:
                in_bin = (probs >= lower) & (probs < upper)

            count = int(in_bin.sum().item())
            if count == 0:
                continue

            mean_conf = probs[in_bin].mean()
            mean_acc = truths[in_bin].mean()
            ece += (count / n) * torch.abs(mean_conf - mean_acc)

        return float(ece.item())

    def metric(self, y_true, y_pred, temperature, raw_probs, calibrated_probs):
        accuracy = metrics.accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        precision = metrics.precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro', zero_division=0)
        recall = metrics.recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro', zero_division=0)
        f1 = metrics.f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro', zero_division=0)
        raw_ce = float(F.binary_cross_entropy(raw_probs.cpu(), y_true.cpu()).item())
        calibrated_ce = float(F.binary_cross_entropy(calibrated_probs.cpu(), y_true.cpu()).item())
        raw_ece = self._compute_quantile_ece(raw_probs.cpu(), y_true.cpu(), bins=self.ece_bins)
        calibrated_ece = self._compute_quantile_ece(calibrated_probs.cpu(), y_true.cpu(), bins=self.ece_bins)

        report_table = {
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1],
            'Raw CE': [raw_ce],
            'Calibrated CE': [calibrated_ce],
            f'Raw ECE@{self.ece_bins}': [raw_ece],
            f'Calibrated ECE@{self.ece_bins}': [calibrated_ece],
            'Temperature': [temperature],
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
        os.makedirs(self.path_metric, exist_ok=True)
        save_path = os.path.join(self.path_metric, f"{self.model_name}.png")
        plt.savefig(save_path)
        plt.close(fig)

        report = {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "raw_cross_entropy": raw_ce,
            "calibrated_cross_entropy": calibrated_ce,
            f"raw_ece_q{self.ece_bins}": raw_ece,
            f"calibrated_ece_q{self.ece_bins}": calibrated_ece,
            "temperature": float(temperature),
            "ece_bins": int(self.ece_bins),
            "metric_plot_path": save_path,
        }

        report_path = os.path.join(self.path_metric, f"{self.model_name}_report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        report["report_path"] = report_path
        return report

    def Get_Y(self):
        val_probs, val_labels = self._collect_binary_outputs(self.val_loader)
        temperature = self._fit_temperature(val_probs, val_labels)

        test_probs, test_labels = self._collect_binary_outputs(self.test_loader)
        calibrated_logits = self._prob_to_logit(test_probs) / temperature
        calibrated_probs = torch.sigmoid(calibrated_logits)

        y_true = test_labels.to('cpu')
        y_pred = (calibrated_probs >= 0.5).float().to('cpu')
        return y_true, y_pred, float(temperature), test_probs.to('cpu'), calibrated_probs.to('cpu')

    def plot_random_heatmap(self):
        # Selecionar um índice aleatório de um lote
        random_index = random.randint(0, min(len(self.test_loader.dataset) - 1, 4562))

        with torch.no_grad():
            matriz, mask, target = self.test_loader.dataset[random_index]

            plt.figure(figsize=(10, 8), dpi=100)  # Ajustar o tamanho e DPI da figura
            plt.imshow(matriz[2], cmap='jet', interpolation='gaussian', aspect='auto')
            plt.colorbar()
            plt.title("Heatmap Aleatório da saída do SoccerMap da bola")
            plt.xlabel("Largura")
            plt.ylabel("Altura")
            os.makedirs(self.path_heatmap, exist_ok=True)
            save_path = os.path.join(self.path_heatmap, f"{self.model_name}_bola.png")
            plt.savefig(save_path)
            plt.close()

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
        y_true, y_pred, temperature, raw_probs, calibrated_probs = self.Get_Y()
        metric_summary = self.metric(
            y_true=y_true,
            y_pred=y_pred,
            temperature=temperature,
            raw_probs=raw_probs,
            calibrated_probs=calibrated_probs,
        )
        self.plot_random_heatmap()
        return metric_summary

@dataclass
class TestConfig:
    model_path: str = "results/models/Pass_success_probability.pt"
    train_directory: str = 'data/processed/pff_match_triplets'
    test_directory: str | None = None
    dataset: PFFDataset | None = None
    model_name: str = "Pass_success_probability"
    path_metric: str = "results/metrics"
    path_heatmap: str = "results/heatmaps"
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Test():
    config = TestConfig()
    if config.dataset is None:
        config.dataset = PFFDataset(
            config.train_directory,
            config.test_directory,
            split_ratio=0.8,
            split_mode="match",
            split_manifest_path="data/processed/cache/splits/pass_match_split.json",
            is_test_mode=True,
        )
    test = TestPassSucess(**config.__dict__)
    test.run()

if __name__ == "__main__":
    Test()
