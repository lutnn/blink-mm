import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from blink_mm.ops.amm_conv2d import AMMConv2d
from blink_mm.ops.amm_linear import AMMLinear

__all__ = ['amm_resnet20', 'amm_resnet32',
           'amm_resnet44', 'amm_resnet56', 'amm_resnet110', 'amm_resnet1202']


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


class Add(nn.Module):
    def forward(self, x, y):
        return x + y


class AMMBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes, planes, stride=1,
        option='A',
        # for ablation study
        temperature_config="inverse", k=16, subvec_len=9, fix_weight=False
    ):
        super().__init__()
        conv1_ncodebooks = in_planes * 9 // subvec_len
        self.conv1 = AMMConv2d(
            conv1_ncodebooks, in_planes, planes,
            kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False,
            temperature_config=temperature_config, k=k, fix_weight=fix_weight)
        self.bn1 = nn.BatchNorm2d(planes)
        conv2_ncodebooks = planes * 9 // subvec_len
        self.conv2 = AMMConv2d(
            conv2_ncodebooks, planes, planes,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
            temperature_config=temperature_config, k=k, fix_weight=fix_weight)
        self.bn2 = nn.BatchNorm2d(planes)

        self.add = Add()
        self.relu = nn.ReLU()

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
        out = self.add(out, self.shortcut(x))
        out = self.relu(out)
        return out


class AMMResNet(nn.Module):
    def __init__(
        self, block, num_blocks, num_classes=10, in_channels=3,
        temperature_config="inverse", k=16, subvec_len=9, fix_weight=False,
        replace_all=False
    ):
        super().__init__()
        self.temperature_config = temperature_config
        self.k = k
        self.subvec_len = subvec_len
        self.fix_weight = fix_weight
        self.in_planes = 16

        self.quantize = nn.Identity()
        if replace_all:
            self.conv1 = AMMConv2d(
                in_channels, in_channels, 16,
                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False,
                temperature_config=temperature_config, k=k, fix_weight=fix_weight
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if replace_all:
            self.linear = AMMLinear(8, 64, num_classes, True, k=k)
        else:
            self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

        if self.fix_weight:
            self.conv1.requires_grad_(False)
            self.linear.requires_grad_(False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride,
                temperature_config=self.temperature_config,
                k=self.k, subvec_len=self.subvec_len,
                fix_weight=self.fix_weight
            ))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.quantize(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, int(out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def amm_resnet20(**kwargs):
    return AMMResNet(AMMBasicBlock, [3, 3, 3], **kwargs)


def amm_resnet32(**kwargs):
    return AMMResNet(AMMBasicBlock, [5, 5, 5], **kwargs)


def amm_resnet44(**kwargs):
    return AMMResNet(AMMBasicBlock, [7, 7, 7], **kwargs)


def amm_resnet56(**kwargs):
    return AMMResNet(AMMBasicBlock, [9, 9, 9], **kwargs)


def amm_resnet110(**kwargs):
    return AMMResNet(AMMBasicBlock, [18, 18, 18], **kwargs)


def amm_resnet1202(**kwargs):
    return AMMResNet(AMMBasicBlock, [200, 200, 200], **kwargs)
