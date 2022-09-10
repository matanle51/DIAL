from collections import OrderedDict

import torch.nn as nn

from .functions import ReverseLayerF


class SmallCNNDANN(nn.Module):

    def __init__(self, drop=0.5):
        super(SmallCNNDANN, self).__init__()
        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2))
        ]))

        self.class_classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, self.num_labels))  # note there is no softmax here
        ]))

        self.domain_classifier = nn.Sequential(OrderedDict([
            ('d_fc1', nn.Linear(64 * 4 * 4, 100)),
            ('d_bn1', nn.BatchNorm1d(100)),
            ('d_relu1', nn.ReLU(True)),
            ('d_fc2', nn.Linear(100, 2))  # Note there is no softmax here
        ]))

    def forward(self, input_data, alpha=None):
        features = self.features(input_data)
        features = features.view(-1, 64 * 4 * 4)
        class_output = self.class_classifier(features)

        if alpha is not None:
            # If alpha is not None we learn the invariant representation
            # Otherwise, in test, we only apply the SmallCNN architecture
            reverse_feature = ReverseLayerF.apply(features, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output

        return class_output
