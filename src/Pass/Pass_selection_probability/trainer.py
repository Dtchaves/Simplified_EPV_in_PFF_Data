import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm
import numpy as np
import logging
import copy
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import DataLoader

try:
    from soccermap import SoccerMapPassSelect,pixel
    from dataloader import PFFDataset
    import utils
except ImportError:
    from .soccermap import SoccerMapPassSelect,pixel
    from .dataloader import PFFDataset
    from . import utils


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _resolve_repo_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)

class TrainerPassSelect:
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
        optim_func,

        model_name,
        path_save_model,
        path_save_loss,

        model,
        data_directory,

    ):

        self.device =  device
        self.epochs = epochs

        self.batch_size = batch_size
        self.batch_sizes = tuple(dict.fromkeys(int(value) for value in (batch_sizes or (batch_size,))))
        self.learning_rate = learning_rate
        self.learning_rates = tuple(dict.fromkeys(float(value) for value in (learning_rates or (learning_rate,))))
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.optim_func = optim_func

        self.model_name = model_name
        self.path_save_model = _resolve_repo_path(path_save_model)
        self.path_save_loss = _resolve_repo_path(path_save_loss)

        self.model = model
        self.data_directory = data_directory

    @staticmethod
    def destination_nll(pred):
        return -torch.log(pred.clamp_min(1e-12)).mean()

    def save_model(self, model):

        os.makedirs(self.path_save_model, exist_ok=True)
        save_path = os.path.join(self.path_save_model, self.model_name + '.pt')

        torch.save(model, save_path)

    def _fit_one(self, dataset, batch_size, learning_rate):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset.get_validation_data(), batch_size=batch_size, shuffle=False)

        model = copy.deepcopy(self.model).to(self.device)
        optim_func = self.optim_func(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )
        best_loss = float("inf")
        best_epoch = -1
        best_state = None
        epochs_without_improvement = 0
        conv_train_losses = []
        conv_val_losses = []

        for t in range(self.epochs):
            train_loss = 0.0
            val_loss = 0.0
            model.train()
            for matriz, mask, target in tqdm(
                train_loader,
                desc=f'TRAINING EPOCH {t}/{self.epochs-1} lr={learning_rate:g} bs={batch_size}',
                dynamic_ncols=True,
                colour="BLUE",
            ):
                matriz = matriz.to(self.device)
                mask = mask.to(self.device).float()

                optim_func.zero_grad()
                surface = model(matriz)
                pred = pixel(surface, mask).view(-1)
                loss = self.destination_nll(pred)
                loss.backward()
                optim_func.step()
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)
            conv_train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                for matriz, mask, target in val_loader:
                    matriz = matriz.to(self.device)
                    mask = mask.to(self.device).float()

                    surface = model(matriz)
                    pred = pixel(surface, mask).view(-1)
                    loss = self.destination_nll(pred)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            conv_val_losses.append(val_loss)

            improved = val_loss < (best_loss - self.early_stopping_delta)
            if improved:
                best_loss = val_loss
                best_epoch = t
                epochs_without_improvement = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                epochs_without_improvement += 1

            logging.info(
                "Grid candidate lr=%s batch_size=%s\nEpoch: %s\nTrain Loss: %s\nValidation Loss: %s\nBest Validation Loss: %s\nBest Epoch: %s\nEpochs Without Improvement: %s\n",
                learning_rate,
                batch_size,
                t,
                train_loss,
                val_loss,
                best_loss,
                best_epoch,
                epochs_without_improvement,
            )

            if epochs_without_improvement >= self.early_stopping_patience:
                logging.info(
                    "Early stopping triggered for lr=%s batch_size=%s at epoch %s after %s epochs without validation improvement greater than %s.",
                    learning_rate,
                    batch_size,
                    t,
                    epochs_without_improvement,
                    self.early_stopping_delta,
                )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, best_loss, conv_train_losses, conv_val_losses, best_epoch


    def run(self):
        dataset = PFFDataset(
            self.data_directory,
            split_ratio=0.8,
            split_mode="match",
            split_manifest_path="data/processed/cache/splits/pass_match_split.json",
        )

        logging.info(
            "\n\n ----- STARTING TRAINING -----\nGrid search learning_rates=%s batch_sizes=%s\n\n",
            self.learning_rates,
            self.batch_sizes,
        )

        best_model = None
        best_loss = float("inf")
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
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = copy.deepcopy(model)
                    best_params = {
                        "learning_rate": float(learning_rate),
                        "batch_size": int(batch_size),
                        "best_epoch": int(best_epoch),
                    }
                    best_train_losses = train_losses
                    best_val_losses = val_losses

        if best_model is None:
            raise RuntimeError("Could not train Pass_selection_probability model.")

        self.save_model(best_model)
        logging.info("Best grid params: %s with validation loss %s", best_params, best_loss)
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
    epochs: int = 100

    batch_size: int = 32
    batch_sizes: tuple[int, ...] = (16, 32)
    learning_rate: float = 1e-5
    learning_rates: tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6)
    weight_decay:float = 0.0
    early_stopping_patience: int = 15
    early_stopping_delta: float = 1e-5
    optim_func:  torch.optim.Adam = field(default_factory=lambda:torch.optim.Adam)


    model_name:str = "Pass_selection_probability"
    path_save_model: str = 'results/models'
    path_save_loss: str = 'results/loss'

    model: SoccerMapPassSelect =  field(default_factory=lambda:SoccerMapPassSelect(in_channels=13))
    data_directory:str = 'data/processed/pff_match_triplets'

def Train():
    logging.basicConfig(level=logging.INFO)
    config = TrainerConfig()
    trainer = TrainerPassSelect(**config.__dict__)
    trainer.run()

if __name__ == "__main__":
    Train()