import torchvision.models as models
import torch
import torch.nn as nn
from models.mlp_head import MLPHead
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



class ByolNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ByolNet, self).__init__()
        
        
        # if kwargs['name'] == 'resnet18':
        #     byolnet = models.resnet18(pretrained=False)
        #     in_channels = byolnet.fc.in_features
        # elif kwargs['name'] == 'resnet50':
        #     byolnet = models.resnet50(pretrained=False)
        #     in_channels = byolnet.fc.in_features
        # elif kwargs['name'] == 'ViT':
        # self.byolnet = models.vision_transformer.vit_b_16(image_size=32, weights=None)#weights='DEFAULT')
        self.byolnet = models.vision_transformer.VisionTransformer(patch_size=4,
                                                                   num_layers=12,
                                                                   num_heads=12,
                                                                   hidden_dim=768,
                                                                   mlp_dim=3072,
                                                                   image_size=32)
        in_channels = self.byolnet.heads.head.in_features

        # self.patch_embedding = torch.nn.Sequential(list(self.byolnet.children())[0])
        # self.encoder = torch.nn.Sequential(*list(self.byolnet.children())[1:-1])
        self.projection = MLPHead(in_channels=in_channels, **kwargs['projection_head'])

    def forward(self, x):
        
        x = self.byolnet._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.byolnet.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.byolnet.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.projection(x)

        return x


    def get_representation(self, x):
        
        x = self.byolnet._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.byolnet.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.byolnet.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return x