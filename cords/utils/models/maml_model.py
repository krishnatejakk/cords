import torch.nn as nn
import torch
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class MAMLNetwork(MetaModule):

    def __init__(self, in_channels, out_features, hidden_size=64):
        super(MAMLNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.features(inputs, params=self.get_subdict(params, 'features'))
                features = features.view((features.size(0), -1))
        else:
            features = self.features(inputs, params=self.get_subdict(params, 'features'))
            features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


class MiniImagenetNetwork(MetaModule):

    def __init__(self, in_channels, out_features, hidden_size=64):
        super(MiniImagenetNetwork, self).__init__()
        embedding_size = 25 * hidden_size
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.classifier = MetaLinear(embedding_size, out_features)

    def forward(self, inputs, params=None, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.features(inputs, params=self.get_subdict(params, 'features'))
                features = features.view((features.size(0), -1))
        else:
            features = self.features(inputs, params=self.get_subdict(params, 'features'))
            features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

def nn_conv3x3(in_channels, out_channels, **kwargs):
    # The convolutional layers (for feature extraction) use standard layers from
    # `torch.nn`, since they do not require adaptation.
    # See `examples/maml/model.py` for comparison.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class FrozenMiniImagenetNetwork(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        self.embedding_size = 25 * hidden_size
        super(FrozenMiniImagenetNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            nn_conv3x3(in_channels, hidden_size),
            nn_conv3x3(hidden_size, hidden_size),
            nn_conv3x3(hidden_size, hidden_size),
            nn_conv3x3(hidden_size, hidden_size)
        )

        # Only the last (linear) layer is used for adaptation in ANIL
        self.classifier = MetaLinear(self.embedding_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits