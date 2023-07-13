import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from blink_mm.ops.quant_amm_conv2d import QuantizedAMMConv2dBatchNorm2dReLU

from qat.ops import *

__all__ = ['quantized_resnet20', 'quantized_resnet32',
           'quantized_resnet44', 'quantized_resnet56', 'quantized_resnet110', 'quantized_resnet1202']


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
        if isinstance(x, QuantizedTensor):
            return x.map(self.lambd)
        else:
            return self.lambd(x)


class QuantizedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', k=16):
        super().__init__()
        self.conv1 = QuantizedAMMConv2dBatchNorm2dReLU(
            in_planes, in_planes, planes,
            kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), activation="relu", k=k
        )
        self.conv2 = QuantizedAMMConv2dBatchNorm2dReLU(
            planes, planes, planes,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), k=k
        )
        self.add = QuantizedAdd()
        self.relu = QuantizedReLU()

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
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.add(out, self.shortcut(x))
        out = self.relu(out)
        return out


class QuantizedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, k=16):
        super().__init__()
        self.in_planes = 16
        self.k = k

        self.quantize = Quantize()
        self.conv1 = QuantizedConv2dBatchNorm2dReLU(
            3, 16, kernel_size=3,
            stride=1, padding=1, bias=False, activation="relu")
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avg_pool = QuantizedAdaptiveAvgPool2d((1, 1))
        self.flatten = QuantizedFlatten(1)
        self.linear = QuantizedLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, k=self.k))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def dequantize(self, x):
        if isinstance(x, QuantizedTensor):
            return x.dequantize()
        else:
            return x

    def forward(self, x):
        out = self.quantize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return self.dequantize(out)


def quantized_resnet20(**kwargs):
    return QuantizedResNet(QuantizedBasicBlock, [3, 3, 3], **kwargs)


def quantized_resnet32(**kwargs):
    return QuantizedResNet(QuantizedBasicBlock, [5, 5, 5], **kwargs)


def quantized_resnet44(**kwargs):
    return QuantizedResNet(QuantizedBasicBlock, [7, 7, 7], **kwargs)


def quantized_resnet56(**kwargs):
    return QuantizedResNet(QuantizedBasicBlock, [9, 9, 9], **kwargs)


def quantized_resnet110(**kwargs):
    return QuantizedResNet(QuantizedBasicBlock, [18, 18, 18], **kwargs)


def quantized_resnet1202(**kwargs):
    return QuantizedResNet(QuantizedBasicBlock, [200, 200, 200])
