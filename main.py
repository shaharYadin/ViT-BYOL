import os

import torch
import yaml
import torchvision.transforms as transforms
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ByolNet
from trainer import BYOLTrainer, ClassifierTrainer
from our_transforms import AddGaussianNoise
from CosineWarmUp import CosineWarmupScheduler
from optuna_exp import objective
import optuna
from define_model import define_model
print(torch.__version__)
torch.manual_seed(0)



def main():

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    tf = transforms.Compose([transforms.ToTensor()])
    

    if config['mode'] == 'optuna':
        trainer = define_model()
        
        # now we can run the experiment
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(study_name="ViT-BYOL-CIFAR10", direction="minimize", sampler=sampler)

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

    elif config['mode'] == 'train_byol':
        trainer = define_model()
        train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=MultiViewDataInjector([tf, tf]))
        trainer.train(train_dataset)

    elif config['mode'] == 'train_classifier':
        train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
        val_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
        
        classifier_trainer = define_model(train_byol=False)
        classifier_trainer.train(train_dataset=train_dataset,val_dataset=val_dataset)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
