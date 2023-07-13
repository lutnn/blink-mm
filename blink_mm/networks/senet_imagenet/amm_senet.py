import torch.nn as nn

from blink_mm.networks.senet_imagenet.senet import SELayer
from blink_mm.networks.resnet_large_imagenet.amm_resnet import AMMResNet, amm_conv3x3

__all__ = ["amm_senet18"]


class AMMSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1,
        base_width=64, dilation=1, norm_layer=None,
        *, reduction=16,
        k=16, subvec_len={"3x3": 9, "1x1": 4},
        temperature_config="inverse",
        fix_weight=False,
    ):
        super().__init__()
        ncodebooks = inplanes * 9 // subvec_len["3x3"]
        self.conv1 = amm_conv3x3(ncodebooks, inplanes, planes, stride, k=k,
                                 temperature_config=temperature_config, fix_weight=fix_weight)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        ncodebooks = planes * 9 // subvec_len["3x3"]
        self.conv2 = amm_conv3x3(
            ncodebooks, planes, planes, 1, k=k, temperature_config="inverse", fix_weight=False)
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


def amm_senet18(**kwargs):
    return AMMResNet(AMMSEBasicBlock, [2, 2, 2, 2], **kwargs)
