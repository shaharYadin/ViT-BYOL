import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import DataLoader
import torch
import os
from models.resnet_base_network import ByolNet
from models.classifier import classifier

def load_weigths(config):
    checkpoint_folder = config['network']['checkpoints']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    byol_model = ByolNet(**config['network']).to(device)
    
    load_params = torch.load(os.path.join(os.path.join(checkpoint_folder, 'byol_model.pth')),
                                            map_location=device)
    byol_model.load_state_dict(load_params['online_network_state_dict'])

    num_classes = 10
    in_channels = byol_model.byolnet.heads.head.out_features
    classifier_model = classifier(in_channels=in_channels,num_classes=num_classes).to(device)
    
    load_params = torch.load(os.path.join(os.path.join(checkpoint_folder, 'classifier_model.pth')),
                                            map_location=device)
    classifier_model.load_state_dict(load_params['classifier_state_dict'])

    return byol_model, classifier_model
                

def classifier_inference(test_data, byol, classifier, sigma=0.1 ,batch_size=256 ,noisy_inference=False):
    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
    )

    byol = byol.to(device)
    classifier = classifier.to(device)
    
    total_images = 0.0
    total_correct = 0.0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            if noisy_inference:
                noise = sigma * torch.randn(imgs.shape).to(device)
                imgs += noise
            imgs = byol.get_representation(imgs)
            outputs = classifier(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_images
    if noisy_inference:
        print(f'The classifier accuracy on the noisy test set is: {accuracy}')
    else:
        print(f'The classifier accutacy on the test set is {accuracy}')
            

            
            