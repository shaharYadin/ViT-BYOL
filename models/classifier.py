from torch import nn


# class classifier(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(classifier, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(in_channels, 2*in_channels),
#             nn.BatchNorm1d(2*in_channels),
#             nn.ReLU(inplace=True),
#         )

#         self.layer2 = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(2*in_channels, 2*in_channels),
#             nn.BatchNorm1d(2*in_channels),
#             nn.ReLU(inplace=True),
#         )

#         self.layer3 = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(2*in_channels, in_channels),
#             nn.BatchNorm1d(in_channels),
#             nn.ReLU(inplace=True),
#         )

#         self.layer4 = nn.Sequential(
#             nn.Linear(in_channels, num_classes)
#         )

#         self.dropout = nn.Dropout
#     def forward(self, x):
#         h = self.layer1(x)
#         h = self.layer2(h)
#         h = self.layer3(h)
#         out = self.layer4(h)
#         return out


class classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(classifier, self).__init__()

        self.linear_classifier = nn.Sequential(
                          nn.BatchNorm1d(in_channels, affine=False),
                          nn.Linear(in_channels, num_classes)
                        )
    
    def forward(self, x):
        return self.linear_classifier(x)

