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
from classifier_inference import classifier_inference,load_weigths
from optuna_exp import objective
import optuna
from define_model import define_model
print(torch.__version__)
torch.manual_seed(0)



def main():

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
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
        train_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=True, download=True, transform=MultiViewDataInjector([tf, tf]))
        trainer.train(train_dataset)

    elif config['mode'] == 'train_classifier':
        dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=True, download=True, transform=tf)

        val_size = int(len(dataset)*0.2)
        train_size = len(dataset) - val_size
        train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        classifier_trainer = define_model(train_byol=False)
        classifier_trainer.train(train_dataset=train_data,val_dataset=valid_data)

    elif config['mode'] == 'test_classifier':
        test_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=False, download=True, transform=tf)
        byol_model, classifier_model = load_weigths(config=config)
        classifier_inference(test_data=test_dataset,byol=byol_model,classifier=classifier_model)
        classifier_inference(test_data=test_dataset,byol=byol_model,classifier=classifier_model,noisy_inference=True)
        

    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
