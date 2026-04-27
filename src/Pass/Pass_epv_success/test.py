"""
Test module for Pass_epv_success model.

Loads trained PE-success model and evaluates it on test data,
computing regression metrics (MSE, MAE) and generating visualization outputs.
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import random
from sklearn.preprocessing import QuantileTransformer
from dataclasses import dataclass, field
from torch.utils.data import DataLoader

from .soccermap import SoccerMapPassEPVSuccess, pixel
from .dataloader import PFFDataset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _resolve_repo_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


class TestPassEPVSuccess:
    def __init__(self, model_path, train_directory, model_name, path_metric, path_heatmap, device, pp_model_path):
        self.model_path = _resolve_repo_path(model_path)
        self.model = torch.load(self.model_path, weights_only=False)
        self.train_directory = train_directory
        resolved_pp_model_path = _resolve_repo_path(pp_model_path)
        self.dataset = PFFDataset(
            train_directory,
            test_directory=None,
            split_ratio=0.8,
            pass_outcome_filter="C",
            pp_model_path=resolved_pp_model_path,
        )
        self.test_loader = DataLoader(self.dataset.get_test_data(), batch_size=32, shuffle=False)
        self.model_name = model_name
        self.path_metric = _resolve_repo_path(path_metric)
        self.path_heatmap = _resolve_repo_path(path_heatmap)
        self.device = device
        self.predictions = []
        self.targets = []

    def compute_metrics(self, y_true, y_pred):
        """Compute regression metrics: MSE, MAE, RMSE."""
        mse = F.mse_loss(y_pred, y_true).item()
        mae = F.l1_loss(y_pred, y_true).item()
        rmse = (mse ** 0.5)

        metrics_dict = {
            'MSE': [mse],
            'MAE': [mae],
            'RMSE': [rmse],
        }

        return metrics_dict

    def save_metrics_table(self, y_true, y_pred):
        """Save metrics as a table visualization."""
        metrics_dict = self.compute_metrics(y_true, y_pred)
        metrics_df = pd.DataFrame(metrics_dict)

        fig, ax = plt.subplots(figsize=(12, 2))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
        table.scale(1, 2)

        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(weight='bold')

        os.makedirs(self.path_metric, exist_ok=True)
        save_path = os.path.join(self.path_metric, f"{self.model_name}_metrics.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        # Also save as CSV
        csv_path = os.path.join(self.path_metric, f"{self.model_name}_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"[Metrics] Saved to {csv_path}")

        return metrics_dict

    def get_predictions(self):
        """Get model predictions on test data."""
        y_true = []
        y_pred = []

        with torch.no_grad():
            for matriz, mask, target in self.test_loader:
                matriz = matriz.to(self.device)
                mask = mask.to(self.device).float()
                target = target.to(self.device).float()

                surface = self.model(matriz)
                pred = pixel(surface, mask).view(-1)

                y_true.append(target)
                y_pred.append(pred)

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)

        return y_true, y_pred

    def plot_sample_heatmaps(self):
        """Plot and save representative heatmaps from test data."""
        random_index = random.randint(0, min(len(self.test_loader.dataset) - 1, 100))

        with torch.no_grad():
            matriz, mask, target = self.test_loader.dataset[random_index]

            os.makedirs(self.path_heatmap, exist_ok=True)

            # Plot input channels
            plt.figure(figsize=(10, 8), dpi=100)
            plt.imshow(matriz[0], cmap='jet', interpolation='gaussian', aspect='auto')
            plt.colorbar()
            plt.title("Input: Attacking Players Heatmap")
            plt.xlabel("Width")
            plt.ylabel("Height")
            save_path = os.path.join(self.path_heatmap, f"{self.model_name}_input_attack.png")
            plt.savefig(save_path)
            plt.close()

            plt.figure(figsize=(10, 8), dpi=100)
            plt.imshow(matriz[1], cmap='jet', interpolation='gaussian', aspect='auto')
            plt.colorbar()
            plt.title("Input: Defending Players Heatmap")
            plt.xlabel("Width")
            plt.ylabel("Height")
            save_path = os.path.join(self.path_heatmap, f"{self.model_name}_input_defense.png")
            plt.savefig(save_path)
            plt.close()

            plt.figure(figsize=(10, 8), dpi=100)
            plt.imshow(matriz[2], cmap='jet', interpolation='gaussian', aspect='auto')
            plt.colorbar()
            plt.title("Input: Ball Heatmap")
            plt.xlabel("Width")
            plt.ylabel("Height")
            save_path = os.path.join(self.path_heatmap, f"{self.model_name}_input_ball.png")
            plt.savefig(save_path)
            plt.close()

            # Plot model output
            matriz_batch = matriz.unsqueeze(0).to(self.device)
            surface = self.model(matriz_batch)
            output_np = surface[0].cpu().detach().numpy().squeeze()

            plt.figure(figsize=(10, 8), dpi=100)
            plt.imshow(output_np, cmap='jet', interpolation='gaussian', aspect='auto')
            plt.colorbar()
            plt.title(f"Model Output: EPV Surface (Target={target:.3f})")
            plt.xlabel("Width")
            plt.ylabel("Height")
            save_path = os.path.join(self.path_heatmap, f"{self.model_name}_output_surface.png")
            plt.savefig(save_path)
            plt.close()

    def plot_prediction_scatter(self, y_true, y_pred):
        """Plot predicted vs actual targets."""
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        plt.figure(figsize=(10, 8), dpi=100)
        plt.scatter(y_true_np, y_pred_np, alpha=0.5, s=20)
        plt.plot([y_true_np.min(), y_true_np.max()], [y_true_np.min(), y_true_np.max()], 'r--', lw=2, label='Perfect prediction')
        plt.xlabel("Target Value")
        plt.ylabel("Predicted Value")
        plt.title(f"{self.model_name}: Prediction vs Target")
        plt.legend()
        plt.grid(True, alpha=0.3)

        os.makedirs(self.path_heatmap, exist_ok=True)
        save_path = os.path.join(self.path_heatmap, f"{self.model_name}_predictions_scatter.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def run(self):
        """Run full evaluation pipeline."""
        self.model.eval()
        print(f"\n[Test] Running PE-success evaluation on {len(self.test_loader.dataset)} samples")

        y_true, y_pred = self.get_predictions()

        # Save metrics
        metrics_dict = self.save_metrics_table(y_true, y_pred)
        print(f"[Metrics] {self.model_name}: MSE={metrics_dict['MSE'][0]:.6f}, MAE={metrics_dict['MAE'][0]:.6f}")

        # Plot heatmaps and scatter
        self.plot_sample_heatmaps()
        self.plot_prediction_scatter(y_true, y_pred)

        print(f"[Test] Evaluation complete. Outputs saved to {self.path_heatmap} and {self.path_metric}")

        return metrics_dict


@dataclass
class TestConfig:
    model_path: str = "results/models/pass_epv_success/Pass_epv_success.pt"
    train_directory: str = "passes"
    model_name: str = "Pass_epv_success"
    path_metric: str = "results/metrics"
    path_heatmap: str = "results/heatmaps"
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pp_model_path: str = "results/models/Pass_success_probability.pt"


def Test():
    config = TestConfig()
    test = TestPassEPVSuccess(**config.__dict__)
    test.run()


if __name__ == "__main__":
    Test()
