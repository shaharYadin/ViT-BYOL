import os

import torch
import yaml
import torchvision.transforms as transforms
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ByolNet
from models.classifier import classifier
from trainer import BYOLTrainer,ClassifierTrainer

print(torch.__version__)
torch.manual_seed(0)


def define_model(train_byol=True):
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
   

    # online network
    online_network = ByolNet(**config['network']).to(device)

    # # load pre-trained model if defined
    # if pretrained_folder:
    #     try:
    #         checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

    #         # load pre-trained parameters
    #         load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
    #                                  map_location=torch.device(torch.device(device)))

    #         online_network.load_state_dict(load_params['online_network_state_dict'])

    #     except FileNotFoundError:
    #         print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ByolNet(**config['network']).to(device)

    optimizer = torch.optim.RAdam(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
    if train_byol:
        trainer = BYOLTrainer(online_network=online_network,
                            target_network=target_network,
                            optimizer=optimizer,
                            predictor=predictor,
                            device=device,
                            **config['trainer'])
        
        return trainer

    else:
        byol_folder = config['network']['checkpoints']
        if byol_folder:
            try:
                load_params = torch.load(os.path.join(os.path.join(byol_folder, 'byol_model.pth')),
                                            map_location=torch.device(torch.device(device)))
                online_network.load_state_dict(load_params['online_network_state_dict'])
                target_network.load_state_dict(load_params['target_network_state_dict'])
                predictor.load_state_dict(load_params['predictor_network_state_dict'])

            except FileNotFoundError:
                print("Pre-trained weights not found.")
                raise FileNotFoundError

        num_classes = 10
        in_channels = online_network.byolnet.heads.head.out_features
        classifier_model = classifier(in_channels=in_channels,num_classes=num_classes)
        classifier_optimizer = torch.optim.Adam(classifier_model.parameters(),
                                **config['optimizer']['classifier_params'])
        
        trainer = ClassifierTrainer(online_network=online_network,
                                    target_network=target_network,
                                    optimizer=classifier_optimizer,
                                    classifier=classifier_model,
                                    predictor=predictor,
                                    device=device,
                                    **config['classifier_trainer'])
        return trainer