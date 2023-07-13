'''MaddnessVGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from blink_mm.ops.maddness.maddness_conv2d import MaddnessConv2d
from blink_mm.networks.vgg_cifar.vgg import new_cfg

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

__all__ = [
    "maddness_vgg11_cifar", "maddness_vgg13_cifar", "maddness_vgg16_cifar", "maddness_vgg19_cifar"
]


class MaddnessVGG(nn.Module):
    def __init__(
        self, vgg_name, subvec_len=9, num_classes=10, in_channels=3,
    ):
        super().__init__()
        self.features = self._make_layers(
            new_cfg[vgg_name], subvec_len, in_channels)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, subvec_len, in_channels):
        layers = []
        first = True
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if first:
                    conv = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                    first = False
                else:
                    ncodebooks = in_channels * 9 // subvec_len
                    conv = MaddnessConv2d(
                        ncodebooks, in_channels, x,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
                    )
                layers += [conv,
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)


def maddness_vgg11_cifar(**kwargs):
    return MaddnessVGG("VGG11", **kwargs)


def maddness_vgg13_cifar(**kwargs):
    return MaddnessVGG("VGG13", **kwargs)


def maddness_vgg16_cifar(**kwargs):
    return MaddnessVGG("VGG16", **kwargs)


def maddness_vgg19_cifar(**kwargs):
    return MaddnessVGG("VGG19", **kwargs)
