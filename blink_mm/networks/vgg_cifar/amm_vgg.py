'''AMMVGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

from blink_mm.ops.amm_conv2d import AMMConv2d
from blink_mm.networks.vgg_cifar.vgg import new_cfg

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

__all__ = [
    "amm_vgg11_cifar", "amm_vgg13_cifar", "amm_vgg16_cifar", "amm_vgg19_cifar"
]


class AMMVGG(nn.Module):
    def __init__(
        self, vgg_name, k=16, subvec_len=9, num_classes=10, in_channels=3,
        temperature_config="inverse", fix_weight=False, replace_all=False
    ):
        super().__init__()
        assert not replace_all
        assert not fix_weight
        assert temperature_config == "inverse"
        self.features = self._make_layers(
            new_cfg[vgg_name], k, subvec_len, in_channels)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, k, subvec_len, in_channels):
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
                    conv = AMMConv2d(
                        ncodebooks, in_channels, x,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True,
                        k=k
                    )
                layers += [conv,
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)


def amm_vgg11_cifar(**kwargs):
    return AMMVGG("VGG11", **kwargs)


def amm_vgg13_cifar(**kwargs):
    return AMMVGG("VGG13", **kwargs)


def amm_vgg16_cifar(**kwargs):
    return AMMVGG("VGG16", **kwargs)


def amm_vgg19_cifar(**kwargs):
    return AMMVGG("VGG19", **kwargs)
