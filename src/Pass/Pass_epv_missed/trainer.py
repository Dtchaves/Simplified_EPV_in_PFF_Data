import logging
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import PFFDataset
from soccermap import SoccerMapPassEPVMissed, pixel


class TrainerPassEPVMissed:
    def __init__(
        self,
        device,
        epochs,
        learning_rate,
        weight_decay,
        loss_func,
        optim_func,
        model_name,
        path_save_model,
        model,
        data_directory,
        batch_size,
    ):
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_func = loss_func
        self.optim_func = optim_func
        self.model_name = model_name
        self.path_save_model = path_save_model
        self.model = model
        self.data_directory = data_directory
        self.batch_size = batch_size

    def save_model(self):
        os.makedirs(self.path_save_model, exist_ok=True)
        save_path = os.path.join(self.path_save_model, self.model_name + ".pt")
        torch.save(self.model, save_path)

    def run(self):
        dataset = PFFDataset(self.data_directory, split_ratio=0.8, pass_outcome_filter="MISSED")
        if len(dataset) == 0:
            raise RuntimeError("No PE-missed training rows available after filtering.")

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset.get_validation_data(), batch_size=self.batch_size, shuffle=False)

        self.model = self.model.to(self.device)
        optimizer = self.optim_func(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_val = float("inf")
        logging.info("\n\n ----- STARTING PE-MISSED TRAINING -----\n\n")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for matrix, mask, target in tqdm(
                train_loader,
                desc=f"TRAINING EPOCH {epoch}/{self.epochs - 1}",
                dynamic_ncols=True,
                colour="BLUE",
            ):
                matrix = matrix.to(self.device)
                mask = mask.to(self.device).float()
                label = target.to(self.device).float()

                optimizer.zero_grad()
                surface = self.model(matrix)
                pred = pixel(surface, mask).view(-1)
                loss = self.loss_func(pred, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for matrix, mask, target in val_loader:
                    matrix = matrix.to(self.device)
                    mask = mask.to(self.device).float()
                    label = target.to(self.device).float()

                    surface = self.model(matrix)
                    pred = pixel(surface, mask).view(-1)
                    loss = self.loss_func(pred, label)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)

            if val_loss < best_val:
                best_val = val_loss
                self.save_model()

            logging.info("Epoch: %s\nTrain Loss: %.6f\nValidation Loss: %.6f\n", epoch, train_loss, val_loss)


@dataclass
class TrainerConfig:
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    loss_func: nn.MSELoss = field(default_factory=lambda: nn.MSELoss())
    optim_func: torch.optim.AdamW = field(default_factory=lambda: torch.optim.AdamW)
    model_name: str = "Pass_epv_missed"
    path_save_model: str = "results/models/pass_epv_missed"
    model: SoccerMapPassEPVMissed = field(default_factory=lambda: SoccerMapPassEPVMissed(in_channels=16))
    data_directory: str = "passes"
    batch_size: int = 32


def Train():
    logging.basicConfig(level=logging.INFO)
    config = TrainerConfig()
    trainer = TrainerPassEPVMissed(**config.__dict__)
    trainer.run()


if __name__ == "__main__":
    Train()
