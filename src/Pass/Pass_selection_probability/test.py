import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn.functional as F
import pandas as pd
import random
from sklearn.preprocessing import QuantileTransformer

from dataclasses import dataclass, field
from torch.utils.data import DataLoader

try:
    from soccermap import SoccerMapPassSelect, pixel
    from dataloader import PFFDataset
except ImportError:
    from .soccermap import SoccerMapPassSelect, pixel
    from .dataloader import PFFDataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _resolve_repo_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)




class TestPassSelect:
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
    def _mask_to_index(mask: torch.Tensor) -> torch.Tensor:
        flat_mask = mask.view(mask.shape[0], -1)
        return flat_mask.argmax(dim=1)

    def _collect_surface_probs_and_labels(self, loader):
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for matriz, mask, target in loader:
                matriz = matriz.to(self.device)
                mask = mask.to(self.device).float()

                surface_probs = self.model(matriz).clamp_min(1e-12)
                label_index = self._mask_to_index(mask)

                all_probs.append(surface_probs)
                all_labels.append(label_index)

        return torch.cat(all_probs), torch.cat(all_labels)

    def _fit_temperature(self, val_probs: torch.Tensor, val_labels: torch.Tensor) -> float:
        logits = torch.log(val_probs.clamp_min(1e-12)).view(val_probs.shape[0], -1).detach()
        labels = val_labels.detach()

        log_temperature = torch.nn.Parameter(torch.zeros(1, device=logits.device))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature).clamp_min(1e-3)
            scaled_logits = logits / temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(torch.exp(log_temperature).detach().item())

    @staticmethod
    def _compute_quantile_ece(probabilities: torch.Tensor, hits: torch.Tensor, bins: int = 10) -> float:
        probs = probabilities.detach().cpu().float().view(-1)
        outcomes = hits.detach().cpu().float().view(-1)
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
            mean_hit = outcomes[in_bin].mean()
            ece += (count / n) * torch.abs(mean_conf - mean_hit)

        return float(ece.item())

    @staticmethod
    def _select_destination_probability(surface_probs: torch.Tensor, label_index: torch.Tensor) -> torch.Tensor:
        flat_surface = surface_probs.view(surface_probs.shape[0], -1)
        return flat_surface.gather(1, label_index.view(-1, 1)).squeeze(1)

    def metric(self, raw_target_probabilities, calibrated_target_probabilities, raw_top1_hits, calibrated_top1_hits, temperature):
        mean_target_probability = float(calibrated_target_probabilities.mean().item())
        mean_nll = float((-torch.log(calibrated_target_probabilities.clamp_min(1e-12))).mean().item())
        top1_accuracy = float(calibrated_top1_hits.mean().item())
        raw_mean_nll = float((-torch.log(raw_target_probabilities.clamp_min(1e-12))).mean().item())
        raw_top1_accuracy = float(raw_top1_hits.mean().item())
        raw_ece = self._compute_quantile_ece(raw_target_probabilities, raw_top1_hits, bins=self.ece_bins)
        calibrated_ece = self._compute_quantile_ece(calibrated_target_probabilities, calibrated_top1_hits, bins=self.ece_bins)
        report_table = {
            'Raw Mean NLL': [raw_mean_nll],
            'Calibrated Mean NLL': [mean_nll],
            'Raw Top-1 Accuracy': [raw_top1_accuracy],
            'Calibrated Top-1 Accuracy': [top1_accuracy],
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
            "mean_target_probability": mean_target_probability,
            "mean_nll": mean_nll,
            "top1_accuracy": top1_accuracy,
            "raw_mean_nll": raw_mean_nll,
            "raw_top1_accuracy": raw_top1_accuracy,
            f"raw_ece_q{self.ece_bins}": raw_ece,
            f"calibrated_ece_q{self.ece_bins}": calibrated_ece,
            "temperature": float(temperature),
            "ece_bins": int(self.ece_bins),
            "sample_count": int(len(calibrated_target_probabilities)),
            "metric_plot_path": save_path,
        }

        report_path = os.path.join(self.path_metric, f"{self.model_name}_report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        report["report_path"] = report_path
        return report

    def Get_Y(self):
        val_probs, val_labels = self._collect_surface_probs_and_labels(self.val_loader)
        temperature = self._fit_temperature(val_probs, val_labels)

        test_probs, test_labels = self._collect_surface_probs_and_labels(self.test_loader)
        test_logits = torch.log(test_probs.clamp_min(1e-12)).view(test_probs.shape[0], -1)
        calibrated_logits = test_logits / temperature
        calibrated_probs_flat = F.softmax(calibrated_logits, dim=1)

        raw_probs_flat = test_probs.view(test_probs.shape[0], -1)
        raw_target_probabilities = raw_probs_flat.gather(1, test_labels.view(-1, 1)).squeeze(1)
        calibrated_target_probabilities = calibrated_probs_flat.gather(1, test_labels.view(-1, 1)).squeeze(1)

        raw_top1 = raw_probs_flat.argmax(dim=1)
        calibrated_top1 = calibrated_probs_flat.argmax(dim=1)
        raw_top1_hits = (raw_top1 == test_labels).float()
        calibrated_top1_hits = (calibrated_top1 == test_labels).float()

        return (
            raw_target_probabilities.to('cpu'),
            calibrated_target_probabilities.to('cpu'),
            raw_top1_hits.to('cpu'),
            calibrated_top1_hits.to('cpu'),
            float(temperature),
        )

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
        (
            raw_target_probabilities,
            calibrated_target_probabilities,
            raw_top1_hits,
            calibrated_top1_hits,
            temperature,
        ) = self.Get_Y()
        metric_summary = self.metric(
            raw_target_probabilities=raw_target_probabilities,
            calibrated_target_probabilities=calibrated_target_probabilities,
            raw_top1_hits=raw_top1_hits,
            calibrated_top1_hits=calibrated_top1_hits,
            temperature=temperature,
        )
        self.plot_random_heatmap()
        return metric_summary

@dataclass
class TestConfig:
    model_path: str = "results/models/Pass_selection_probability.pt"
    train_directory: str = 'data/processed/pff_match_triplets'
    test_directory: str = None
    dataset: PFFDataset | None = None
    model_name: str = "Pass_selection_probability"
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
    test = TestPassSelect(**config.__dict__)
    return test.run()

if __name__ == "__main__":
    Test()
