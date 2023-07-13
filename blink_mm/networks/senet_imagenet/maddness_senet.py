import torch.nn as nn

from blink_mm.networks.senet_imagenet.senet import SELayer
from blink_mm.networks.resnet_large_imagenet.maddness_resnet import MaddnessResNet, maddness_conv3x3

__all__ = ["maddness_senet18"]


class MaddnessSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=None,
        *, reduction=16,
        subvec_len={"3x3": 9, "1x1": 4},
    ):
        super().__init__()
        ncodebooks = inplanes * 9 // subvec_len["3x3"]
        self.conv1 = maddness_conv3x3(ncodebooks, inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        ncodebooks = planes * 9 // subvec_len["3x3"]
        self.conv2 = maddness_conv3x3(
            ncodebooks, planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def maddness_senet18(**kwargs):
    return MaddnessResNet(MaddnessSEBasicBlock, [2, 2, 2, 2], **kwargs)
