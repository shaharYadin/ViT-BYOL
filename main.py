import os

import optuna
import torch
import torchvision.transforms as transforms
import wandb
import yaml
from torchvision import datasets

from check_similarity_per_layer import check_similarity_per_layer
from classifier_inference import classifier_inference, load_weigths, save_imgs
from CosineWarmUp import CosineWarmupScheduler
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from define_model import define_model
from models.mlp_head import MLPHead
from models.resnet_base_network import ByolNet
from optuna_exp import objective
from our_transforms import AddGaussianNoise
from trainer import BYOLTrainer, ClassifierTrainer

print(torch.__version__)
# torch.manual_seed(0)

wandb.init(project="vit_byol", sync_tensorboard=True, entity='vit_byol')

def main():

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    sigma = config['trainer']['sigma']
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running in mode {config['mode']}")
    if config['network']['pretrained']:
        tf = transforms.Compose([transforms.Resize(size=(224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        noisy_tf = transforms.Compose([transforms.Resize(size=(224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                 AddGaussianNoise(std=sigma)])
    else:
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
        noisy_tf = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                       AddGaussianNoise(std=sigma)])


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
        dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=True, download=True, transform=MultiViewDataInjector([tf, tf]))
        val_size = int(len(dataset)*0.2)
        train_size = len(dataset) - val_size
        train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, val_size])

        trainer.train(train_data, valid_data)

    elif config['mode'] == 'train_classifier':
        dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=True, download=True, transform=tf)

        val_size = int(len(dataset)*0.2)
        train_size = len(dataset) - val_size
        train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        classifier_trainer = define_model(train_byol=False)
        classifier_trainer.train(train_dataset=train_data,val_dataset=valid_data)

    elif config['mode'] == 'test_classifier':
        
        test_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=False, download=True, transform=tf)
        # noisy_test_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=False, download=True, transform=tf)
        byol_model, classifier_model, byol_time, classifier_time = load_weigths(config=config)
        classifier_inference(test_data=test_dataset, byol=byol_model,classifier=classifier_model, byol_time=byol_time, classifier_time=classifier_time)
        classifier_inference(test_data=test_dataset, byol=byol_model,classifier=classifier_model, byol_time=byol_time, classifier_time=classifier_time , noisy_inference=True, sigma=sigma)

        test_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=False, download=True, transform=MultiViewDataInjector([tf, tf]))
        save_imgs(test_dataset, byol_model, classifier_model, sigma=sigma)
    
    elif config['mode'] == 'check_similarity_per_layer':
        byol_model, _, _, _ = load_weigths(config=config)
        test_dataset = datasets.CIFAR10("/tmp/ramdisk/data/", train=False, download=True, transform=MultiViewDataInjector([tf, tf])) 
        
        check_similarity_per_layer(byol_model, test_dataset, sigma=sigma)
        
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
