import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import numpy as np

from models.classifier import classifier
from models.resnet_base_network import ByolNet


def load_weigths(config):
    checkpoint_folder = config['network']['checkpoints']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    byol_model = ByolNet(**config['network']).to(device)
    
    load_params = torch.load(os.path.join(checkpoint_folder, 'byol_model.pth'),
                                            map_location=device)
    byol_model.load_state_dict(load_params['online_network_state_dict'])

    num_classes = 10
    in_channels = byol_model.byolnet.heads.head.in_features
    classifier_model = classifier(in_channels=in_channels,num_classes=num_classes).to(device)
    
    load_params = torch.load(os.path.join(checkpoint_folder, 'classifier_model.pth'),
                                            map_location=device)
    classifier_model.load_state_dict(load_params['classifier_state_dict'])

    time_byol = os.path.getmtime(os.path.join(checkpoint_folder, 'byol_model.pth'))
    time_classifier = os.path.getmtime(os.path.join(checkpoint_folder, 'classifier_model.pth'))

    return byol_model, classifier_model, time_byol, time_classifier
                

def classifier_inference(test_data, byol, classifier, byol_time, classifier_time ,sigma=0.1 ,batch_size=256 ,noisy_inference=False):
    

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
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            # if noisy_inference:
            #     noise = sigma * torch.randn(imgs.shape).to(device)
            #     imgs += noise
            imgs = byol.get_representation(imgs)
            imgs = F.normalize(imgs, dim=1)
            outputs = classifier(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_images

    byol_time = time.ctime(byol_time)
    classifier_time = time.ctime(classifier_time)
    if noisy_inference:
        print(f'The classifier accuracy on the noisy test set is: {accuracy}')
        with open('inference.txt', 'a') as file1:
            file1.write(f'BYOL was created at {byol_time}, Classifier was created at {classifier_time}:\n Noisy test set Accuracy: {accuracy}\n')
    else:
        print(f'The classifier accuracy on the test set is {accuracy}')
        with open('inference.txt', 'a') as file1:
            file1.write(f'BYOL was created at {byol_time}, Classifier was created at {classifier_time}:\n Test set Accuracy: {accuracy}\n')
            

def save_imgs(test_data, byol, classifier ,batch_size=8):            
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
    )

    byol = byol.to(device)
    classifier = classifier.to(device)
    
    (noisy_imgs, imgs), labels = next(iter(test_loader))
    with torch.no_grad():
        noisy_imgs = noisy_imgs.to(device) #+ torch.randn(noisy_imgs.shape).to(device)
        imgs= imgs.to(device)
        labels = labels.to(device)
        
        imgs_representation = byol.get_representation(imgs)
        imgs = F.normalize(imgs, dim=1)
        outputs = classifier(imgs_representation)
        _, predicted = torch.max(outputs.data, 1)

        noisy_imgs_representation = byol.get_representation(noisy_imgs)
        noisy_imgs = F.normalize(noisy_imgs, dim=1)
        noisy_outputs = classifier(noisy_imgs_representation)
        _, noisy_predicted = torch.max(noisy_outputs.data, 1)
    
    noisy_imgs = (noisy_imgs / 2  + 0.5).clamp(0.0, 1.0)
    imgs = (imgs / 2 + 0.5).clamp(0.0, 1.0)

    labels_to_classes = {0: "airplane",
                        1: "automobile",
                        2:"bird",
                        3:"cat",
                        4:"deer",
                        5:"dog",
                        6:"frog",
                        7:"horse",
                        8:"ship",
                        9:"truck"
                        }
    fig = plt.figure(figsize=(20,20))
    for i in range(batch_size):
        ax = plt.subplot(4,4, 2*i+1)
        ax.imshow(np.array(imgs[i].cpu()).transpose(1,2,0))
        ax.set_title(f'Predicted Label: {labels_to_classes[predicted[i].item()]}')
        ax2 = plt.subplot(4,4, 2*i+2)
        ax2.imshow(np.array(noisy_imgs[i].cpu()).transpose(1,2,0))
        ax2.set_title(f'Predicted Label: {labels_to_classes[noisy_predicted[i].item()]}')
    
    fig.savefig('images_with_predictions.png')

