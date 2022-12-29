import optuna
import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from data.multi_view_data_injector import MultiViewDataInjector

from utils import _create_model_training_folder
from CosineWarmUp import CosineWarmupScheduler
from trainer import BYOLTrainer
import torchvision.transforms as transforms
from define_model import define_model

def get_cifar10(batch_size):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./datasets', train=True, download=True, transform=MultiViewDataInjector([transforms.ToTensor(), transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('./datasets', train=False, transform=MultiViewDataInjector([transforms.ToTensor(), transforms.ToTensor()])),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, valid_loader

def objective(trial,n_train_batches=30,n_valid_batches=10):
    
    trainer = define_model()
    trainer.max_epochs = 50
    # Generate the model.
    model = trainer.online_network.to(trainer.device)
    model_predictor = trainer.predictor.to(trainer.device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  # log=True, will use log scale to interplolate between lr
    trainer.optimizer = torch.optim.RAdam(list(model.parameters()) + list(model_predictor.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # trainer.optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    # alternative version
    # optimizer = trial.suggest_categorical("optimizer", [optim.Adam, optim.RMSprop, optim.SGD])

    # Get the CIFAR10 dataset.
    train_loader, valid_loader = get_cifar10(batch_size=trainer.batch_size)


    # Training of the model.
    for epoch in range(trainer.max_epochs):
        model.train()  
        model_predictor.train()
        n_iter = 0
        for (batch_view_1, batch_view_2), _ in train_loader:
            # Limiting training data for faster epochs.
            if n_iter  >= n_train_batches:
                break

            batch_view_1 = batch_view_1.to(trainer.device)
            batch_view_2 = batch_view_2.to(trainer.device)
            
            sigma = 0.1
            noise = (sigma * torch.randn(batch_view_1.shape)).to(trainer.device)
            batch_view_1 += noise

            loss = trainer.update(batch_view_1, batch_view_2)

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            trainer._update_target_network_parameters()  # update the key encoder
            n_iter += 1
        
        # Validation of the model.
        model.eval()
        model_predictor.eval()
        val_loss = 0
        val_niter = 0
        with torch.no_grad():
            for (batch_view_1, batch_view_2), _ in valid_loader:
                # Limiting validation data.
                if val_niter  >= n_valid_batches:
                    break

                batch_view_1 = batch_view_1.to(trainer.device)
                batch_view_2 = batch_view_2.to(trainer.device)
                
                sigma = 0.1
                noise = (sigma * torch.randn(batch_view_1.shape)).to(trainer.device)

                batch_view_1 += noise
                val_loss += (trainer.update(batch_view_1, batch_view_2)).item()
                val_niter += 1
        val_loss = val_loss / val_niter

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(val_loss, epoch)  

        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

