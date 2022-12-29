import os

import torch
import yaml
import torchvision.transforms as transforms
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ByolNet
from trainer import BYOLTrainer
from our_transforms import AddGaussianNoise
from CosineWarmUp import CosineWarmupScheduler
from optuna_exp import objective
import optuna
from define_model import define_model
print(torch.__version__)
torch.manual_seed(0)



def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    tf = transforms.Compose([transforms.ToTensor()])
                                      
    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=MultiViewDataInjector([tf, tf]))
    trainer = define_model()

    if (config['optuna']):
        # now we can run the experiment
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(study_name="ViT-BYOL-CIFAR10", direction="minimize", sampler=sampler)
        # func = lambda trial: objective(trial, trainer=trainer)

        study.optimize(objective, n_trials=30)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    else:
        trainer.train(train_dataset)


if __name__ == '__main__':
    main()
