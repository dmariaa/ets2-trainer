from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        up_x = self.convA(torch.cat([up_x, concat_with], dim=1))
        up_x = self.leakyreluA(up_x)
        up_x = self.convB(up_x)
        up_x = self.leakyreluB(up_x)
        return up_x


class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width=1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features // 1 + 256, output_features=features // 2)
        self.up2 = UpSample(skip_input=features // 2 + 128, output_features=features // 4)
        self.up3 = UpSample(skip_input=features // 4 + 64, output_features=features // 8)
        self.up4 = UpSample(skip_input=features // 8 + 64, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, input, skip_features):
        x_d0 = self.conv2(F.relu(skip_features['norm5']))
        x_d1 = self.up1(x_d0, skip_features['transition2'])
        x_d2 = self.up2(x_d1, skip_features['transition1'])
        x_d3 = self.up3(x_d2, skip_features['pool0'])
        x_d4 = self.up4(x_d3, skip_features['relu0'])
        return self.conv3(x_d4)


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        print(f"Loading densenet pretrained={pretrained}")
        self.densenet = models.densenet169(pretrained=pretrained).features
        self.densenet_features = OrderedDict()
        self.fhooks = []
        self.output_layers = ['norm5', 'transition2', 'transition1', 'pool0', 'relu0']

        for i, l in enumerate(list(self.densenet._modules.keys())):
            if l in self.output_layers:
                self.fhooks.append(getattr(self.densenet, l).register_forward_hook(self.forward_hook(l)))

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.densenet = self.densenet.to(*args, **kwargs)
        print(f"densenet sent to {args}")
        return self

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.densenet_features[layer_name] = output

        return hook

    def forward(self, x):
        output = self.densenet(x)
        return output, self.densenet_features


class Model(nn.Module):
    def __init__(self, pretrained_encoder=True):
        super(Model, self).__init__()
        self.encoder = Encoder(pretrained_encoder)
        self.decoder = Decoder()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        return self

    def forward(self, x):
        x, skip_features = self.encoder(x)
        x = self.decoder(x, skip_features)
        return x
