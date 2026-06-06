import logging
import os
import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from dataloader import PFFDataset
    from soccermap import SoccerMapPassEPVSuccess, pixel
    import utils
except ImportError:
    from .dataloader import PFFDataset
    from .soccermap import SoccerMapPassEPVSuccess, pixel
    from . import utils


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


class TrainerPassEPVSuccess:
    def __init__(
        self,
        device,
        epochs,
        batch_size,
        batch_sizes,
        learning_rate,
        learning_rates,
        weight_decay,
        early_stopping_patience,
        early_stopping_delta,
        loss_func,
        optim_func,
        model_name,
        path_save_model,
        path_save_loss,
        model,
        data_directory,
        pp_model_path,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_sizes = tuple(dict.fromkeys(int(value) for value in (batch_sizes or (batch_size,))))
        self.learning_rate = learning_rate
        self.learning_rates = tuple(dict.fromkeys(float(value) for value in (learning_rates or (learning_rate,))))
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.loss_func = loss_func
        self.optim_func = optim_func
        self.model_name = model_name
        self.path_save_model = _resolve_repo_path(path_save_model)
        self.model = model
        self.data_directory = data_directory
        self.pp_model_path = _resolve_repo_path(pp_model_path)
        self.path_save_loss = _resolve_repo_path(path_save_loss)

    def save_model(self):
        os.makedirs(self.path_save_model, exist_ok=True)
        save_path = os.path.join(self.path_save_model, self.model_name + ".pt")
        torch.save(self.model, save_path)

    def _save_model(self, model):
        os.makedirs(self.path_save_model, exist_ok=True)
        save_path = os.path.join(self.path_save_model, self.model_name + ".pt")
        torch.save(model, save_path)

    def _fit_one(self, dataset, batch_size, learning_rate):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset.get_validation_data(), batch_size=batch_size, shuffle=False)

        model = copy.deepcopy(self.model).to(self.device)
        optimizer = self.optim_func(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )

        best_val = float("inf")
        best_epoch = -1
        best_state = None
        epochs_without_improvement = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0.0
            for matrix, mask, target in tqdm(
                train_loader,
                desc=f"TRAINING EPOCH {epoch}/{self.epochs - 1} lr={learning_rate:g} bs={batch_size}",
                dynamic_ncols=True,
                colour="BLUE",
            ):
                matrix = matrix.to(self.device)
                mask = mask.to(self.device).float()
                label = target.to(self.device).float()

                optimizer.zero_grad()
                surface = model(matrix)
                pred = pixel(surface, mask).view(-1)
                loss = self.loss_func(pred, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for matrix, mask, target in val_loader:
                    matrix = matrix.to(self.device)
                    mask = mask.to(self.device).float()
                    label = target.to(self.device).float()

                    surface = model(matrix)
                    pred = pixel(surface, mask).view(-1)
                    loss = self.loss_func(pred, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)

            improved = val_loss < (best_val - self.early_stopping_delta)
            if improved:
                best_val = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            logging.info(
                "Grid candidate lr=%s batch_size=%s\nEpoch: %s\nTrain Loss: %.6f\nValidation Loss: %.6f\nBest Validation Loss: %.6f\nBest Epoch: %s\nEpochs Without Improvement: %s\n",
                learning_rate,
                batch_size,
                epoch,
                train_loss,
                val_loss,
                best_val,
                best_epoch,
                epochs_without_improvement,
            )

            if epochs_without_improvement >= self.early_stopping_patience:
                logging.info(
                    "Early stopping triggered for lr=%s batch_size=%s at epoch %s after %s epochs without validation improvement greater than %.1e.",
                    learning_rate,
                    batch_size,
                    epoch,
                    epochs_without_improvement,
                    self.early_stopping_delta,
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, best_val, train_losses, val_losses, best_epoch

    def run(self):
        dataset = PFFDataset(
            self.data_directory,
            split_ratio=0.8,
            split_mode="match",
            split_manifest_path="data/processed/cache/splits/pass_match_split.json",
            pass_outcome_filter="C",
            pp_model_path=self.pp_model_path,
        )
        if len(dataset) == 0:
            raise RuntimeError("No PE-success training rows available after filtering.")

        logging.info(
            "\n\n ----- STARTING PE-SUCCESS TRAINING -----\nGrid search learning_rates=%s batch_sizes=%s\n\n",
            self.learning_rates,
            self.batch_sizes,
        )

        best_model = None
        best_val = float("inf")
        best_params = None
        best_train_losses = []
        best_val_losses = []

        for learning_rate in self.learning_rates:
            for batch_size in self.batch_sizes:
                logging.info("Training grid candidate learning_rate=%s batch_size=%s", learning_rate, batch_size)
                model, val_loss, train_losses, val_losses, best_epoch = self._fit_one(
                    dataset,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
                if val_loss < best_val:
                    best_val = val_loss
                    best_model = copy.deepcopy(model)
                    best_params = {
                        "learning_rate": float(learning_rate),
                        "batch_size": int(batch_size),
                        "best_epoch": int(best_epoch),
                    }
                    best_train_losses = train_losses
                    best_val_losses = val_losses

        if best_model is None:
            raise RuntimeError("Could not train Pass_epv_success model.")

        self._save_model(best_model)
        logging.info("Best grid params: %s with validation loss %.6f", best_params, best_val)
        if best_train_losses and len(best_train_losses) > 1:
            os.makedirs(self.path_save_loss, exist_ok=True)
            utils.plot_loss(
                best_train_losses,
                best_val_losses,
                best_params["best_epoch"],
                self.model_name,
                self.path_save_loss,
            )


@dataclass
class TrainerConfig:
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs: int = 20
    batch_size: int = 16
    batch_sizes: tuple[int, ...] = (16, 32)
    learning_rate: float = 0.001
    learning_rates: tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6)
    weight_decay: float = 0.0
    early_stopping_patience: int = 6
    early_stopping_delta: float = 1e-5
    loss_func: nn.MSELoss = field(default_factory=lambda: nn.MSELoss())
    optim_func: torch.optim.Adam = field(default_factory=lambda: torch.optim.Adam)
    model_name: str = "Pass_epv_success"
    path_save_model: str = "results/models/pass_epv_success"
    model: SoccerMapPassEPVSuccess = field(default_factory=lambda: SoccerMapPassEPVSuccess(in_channels=16))
    data_directory: str = "data/processed/pff_match_triplets"
    pp_model_path: str = "results/models/Pass_success_probability.pt"
    path_save_loss: str = 'results/loss/pass_epv_success'


def Train():
    logging.basicConfig(level=logging.INFO)
    config = TrainerConfig()
    trainer = TrainerPassEPVSuccess(**config.__dict__)
    trainer.run()


if __name__ == "__main__":
    Train()
