import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from blink_mm.ops.maddness.maddness_conv2d import MaddnessConv2d
from blink_mm.ops.maddness.maddness_linear import MaddnessLinear

__all__ = ['maddness_resnet20', 'maddness_resnet32',
           'maddness_resnet44', 'maddness_resnet56', 'maddness_resnet110', 'maddness_resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class MaddnessBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.conv1 = MaddnessConv2d(
            in_planes, in_planes, planes,
            kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaddnessConv2d(
            planes, planes, planes,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MaddnessResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, replace_linear=False):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if replace_linear:
            self.linear = MaddnessLinear(8, 64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, int(out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def maddness_resnet20(**kwargs):
    return MaddnessResNet(MaddnessBasicBlock, [3, 3, 3], **kwargs)


def maddness_resnet32(**kwargs):
    return MaddnessResNet(MaddnessBasicBlock, [5, 5, 5], **kwargs)


def maddness_resnet44(**kwargs):
    return MaddnessResNet(MaddnessBasicBlock, [7, 7, 7], **kwargs)


def maddness_resnet56(**kwargs):
    return MaddnessResNet(MaddnessBasicBlock, [9, 9, 9], **kwargs)


def maddness_resnet110(**kwargs):
    return MaddnessResNet(MaddnessBasicBlock, [18, 18, 18], **kwargs)


def maddness_resnet1202(**kwargs):
    return MaddnessResNet(MaddnessBasicBlock, [200, 200, 200])
