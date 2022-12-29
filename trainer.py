import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import _create_model_training_folder
from CosineWarmUp import CosineWarmupScheduler


class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        # Warm up schedular since we use ViT
        # warmup = 3 * len(train_loader)
        # max_iter = 50 * len(train_loader)
        # scheduler = CosineWarmupScheduler(self.optimizer, warmup=warmup, max_iters=max_iter)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        best_loss = np.inf
        for epoch_counter in range(self.max_epochs):
            epoch_loss = 0
            for (batch_view_1, batch_view_2), _ in train_loader:

                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)
                
                sigma = 0.1
                noise = (sigma * torch.randn(batch_view_1.shape)).to(self.device)
                batch_view_1 += noise

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1
                # scheduler.step()
            if epoch_loss < best_loss:
                self.save_model(os.path.join(model_checkpoints_folder, 'byol_model.pth'))
                best_loss = epoch_loss

            print("End of epoch {}".format(epoch_counter))

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'byol_model_final.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        # predictions_from_view_2 = self.predictor(self.online_network(batch_view_2)) 

        # compute key features
        with torch.no_grad():
            # targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        # loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'predictor_network_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)


class ClassifierTrainer:
    def __init__(self, online_network, target_network, classifier, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.classifier = classifier
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])


    def train(self, train_dataset, val_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        val_niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.target_network.to(self.device)
        self.classifier.to(self.device)

        self.target_network.eval()
        for epoch_counter in range(self.max_epochs):

            self.classifier.train()
            for images, labels in train_loader:
                
                loss = self.update(images, labels)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                niter += 1
            
            if epoch_counter + 1 % 5 == 0:
                self.classifier.eval()
                with torch.no_grad():
                    for images, labels in val_loader:
                        val_loss = self.update(images, labels)
                        self.writer.add_scalar('val_loss', val_loss, global_step=val_niter)
                        val_niter += 1
        
            print(f'End of epoch {epoch_counter}, val_loss={val_loss}')

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'classifier_model.pth'))

    def update(self, images, labels):
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        with torch.no_grad():
            images_embedding = self.target_network(images)
                
        outputs = self.classifier(images_embedding)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        return loss

    def save_model(self, PATH):
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
        }, PATH)