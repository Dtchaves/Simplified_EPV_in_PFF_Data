import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import logging
from dataclasses import dataclass, field
from typing import Any

from soccermap import SoccerMapPassSucess,pixel
from torch.utils.data import DataLoader
from dataloader import PFFDataset
import utils

class TrainerPassSucess:
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
        path_save_loss,
        
        model,
        data_directory,
        
    ):
        
        self.device =  device
        self.epochs = epochs
            
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_func = loss_func
        self.optim_func = optim_func
            
        self.model_name = model_name
        self.path_save_model = path_save_model
        self.path_save_loss = path_save_loss
        
        self.model = model
        self.data_directory = data_directory

        
    def save_models(self,ckp,t):

        save_path = os.path.join(self.path_save_model, self.model_name + '.pt')
            
        torch.save(self.model,save_path)
        
    
    def run(self):
        
        dataset = PFFDataset(self.data_directory, split_ratio=0.8)

        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset.get_validation_data(), batch_size=32, shuffle=False)
                
        self.model = self.model.to(self.device) 
        loss_func = self.loss_func 
        
        optim_func = self.optim_func(self.model.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
        best_loss = 1e12
        conv_train_losses = []
        conv_val_losses = []
        
        logging.info("\n\n ----- STARTING TRAINING -----\n\n")
        
        for t in range(self.epochs):
            
            train_loss = 0.0
            val_loss = 0.0
            for matriz, mask, target in tqdm(train_loader, desc=f'TRAINING EPOCH {t}/{self.epochs-1}',dynamic_ncols=True,colour="BLUE",):
                
                matriz = matriz.to(self.device)
                mask = mask.to(self.device).float()
                label = target.to(self.device).float()
                
    
                optim_func.zero_grad()
                
                surface = self.model(matriz)
                pred = pixel(surface, mask).view(-1)
                loss = loss_func(pred, label)
                loss.backward()
                optim_func.step()
                
                
                train_loss += loss.item()
                                
            train_loss = train_loss/len(train_loader)
            conv_train_losses.append(train_loss)
            
            with torch.no_grad():
                for matriz, mask, target in val_loader:
                    
                    matriz = matriz.to(self.device)
                    mask = mask.to(self.device).float()
                    label = target.to(self.device).float()
                    
        
                    optim_func.zero_grad()
                    
                    surface = self.model(matriz)
                    pred = pixel(surface, mask).view(-1)
                    loss = loss_func(pred, label)

                    
                    
                    val_loss += loss.item()
                
            val_loss = val_loss / len(val_loader)
            conv_val_losses.append(val_loss)
            
            if best_loss > val_loss:
                best_loss = val_loss
                self.save_models(ckp=False,t=t)
                
                
            #if t % 10 == 0:
            logging.info(f"Epoch: {t}\nTrain Loss: {train_loss}\nValidation Loss: {val_loss}\n")
            if t != 0:
                utils.plot_loss(conv_train_losses,conv_val_losses,t,self.model_name,self.path_save_loss)         
                    
                    
        
@dataclass
class TrainerConfig:
    
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs: int = 100
    
    learning_rate: float = 0.01
    weight_decay:float = 0.01
    loss_func:  nn.BCELoss = field(default_factory=lambda:nn.BCELoss())
    optim_func:  torch.optim.AdamW = field(default_factory=lambda:torch.optim.AdamW)
    
    
    model_name:str = "Pass_success_probability"
    path_save_model: str = '/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/results/models'
    path_save_loss: str = '/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/results/loss'
    
    model: SoccerMapPassSucess =  field(default_factory=lambda:SoccerMapPassSucess(in_channels=17))    
    data_directory:str = '/home_cerberus/disk2/diogochaves/FUTEBOL/Simplified_EPV_in_PFF_Data/data/Pass'

def Train():
    logging.basicConfig(level=logging.INFO) 
    config = TrainerConfig()
    trainer = TrainerPassSucess(**config.__dict__)
    trainer.run()

if __name__ == "__main__":
    Train()