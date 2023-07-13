import torch
import torch.nn as nn
import torch.nn.functional as F

from blink_mm.ops.amm_conv2d import AMMConv2d

__all__ = ["amm_senet18_cifar"]


class AMMPreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, k=16, subvec_len={"3x3": 9, "1x1": 4}):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        ncodebooks = in_planes * 9 // subvec_len["3x3"]
        self.conv1 = AMMConv2d(
            ncodebooks, in_planes, planes, kernel_size=(3, 3),
            stride=(stride, stride), padding=(1, 1), bias=False, k=k
        )
        self.bn2 = nn.BatchNorm2d(planes)
        ncodebooks = planes * 9 // subvec_len["3x3"]
        self.conv2 = AMMConv2d(
            ncodebooks, planes, planes, kernel_size=(3, 3),
            stride=(1, 1), padding=(1, 1), bias=False, k=k
        )

        if stride != 1 or in_planes != planes:
            ncodebooks = in_planes // subvec_len["1x1"]
            self.shortcut = nn.Sequential(
                AMMConv2d(ncodebooks, in_planes, planes, kernel_size=(1, 1),
                          stride=(stride, stride), bias=False, k=k)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, int(out.size(2)))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class AMMSENet(nn.Module):
    def __init__(
        self, block, num_blocks, k=16, subvec_len={"3x3": 9, "1x1": 4}, num_classes=10, in_channels=3,
        temperature_config="inverse", fix_weight=False, replace_all=False
    ):
        super().__init__()

        self.k = k
        self.subvec_len = subvec_len
        assert temperature_config == "inverse"
        assert not fix_weight
        assert not replace_all

        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_planes, planes, stride,
                k=self.k, subvec_len=self.subvec_len
            ))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def amm_senet18_cifar(**kwargs):
    return AMMSENet(AMMPreActBlock, [2, 2, 2, 2], **kwargs)
