from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from blink_mm.networks.resnet_small_cifar.amm_resnet import Add
from blink_mm.ops.amm_conv2d import AMMConv2d


__all__ = ['amm_resnet18_cifar', 'amm_resnet34_cifar',
           'amm_resnet50_cifar', 'amm_resnet101_cifar', 'amm_resnet152_cifar']


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def amm_conv3x3(ncodebooks, in_planes: int, out_planes: int, stride: int = 1, k=16, temperature_config="inverse", fix_weight=False):
    return AMMConv2d(
        ncodebooks,
        in_planes,
        out_planes,
        (3, 3),
        (stride, stride),
        (1, 1),
        bias=False,
        k=k,
        temperature_config=temperature_config,
        fix_weight=fix_weight
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def amm_conv1x1(ncodebooks, in_planes: int, out_planes: int, stride: int = 1, k=16, temperature_config="inverse", fix_weight=False):
    return AMMConv2d(
        ncodebooks,
        in_planes,
        out_planes,
        (1, 1),
        (stride, stride),
        bias=False,
        k=k,
        temperature_config=temperature_config,
        fix_weight=fix_weight
    )


class AMMBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        k=16,
        subvec_len={"3x3": 9, "1x1": 4},
        temperature_config="inverse",
        fix_weight=False
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = amm_conv3x3(
            inplanes * 9 // subvec_len["3x3"], inplanes, planes, stride, k=k, temperature_config=temperature_config, fix_weight=fix_weight)
        self.bn1 = norm_layer(planes)
        self.conv2 = amm_conv3x3(
            planes * 9 // subvec_len["3x3"], planes, planes, k=k, temperature_config=temperature_config, fix_weight=fix_weight)
        self.bn2 = norm_layer(planes)
        self.add = Add()
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class AMMBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        k=16,
        subvec_len={"3x3": 9, "1x1": 4},
        temperature_config="inverse",
        fix_weight=False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = amm_conv1x1(
            inplanes // subvec_len["1x1"], inplanes, width,
            k=k, temperature_config=temperature_config, fix_weight=fix_weight
        )
        self.bn1 = norm_layer(width)
        self.conv2 = amm_conv3x3(
            width * 9 // subvec_len["3x3"], width, width, stride, k=k,
            temperature_config=temperature_config, fix_weight=fix_weight
        )
        self.bn2 = norm_layer(width)
        self.conv3 = amm_conv1x1(
            width // subvec_len["1x1"], width, planes * self.expansion, k=k,
            temperature_config=temperature_config, fix_weight=fix_weight
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.add = Add()
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class AMMResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[AMMBasicBlock, AMMBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_channels=3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        k=16,
        subvec_len={"3x3": 9, "1x1": 4},
        temperature_config="inverse",
        fix_weight=False,
        replace_all=False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.k = k
        self.subvec_len = subvec_len
        self.temperature_config = temperature_config
        self.fix_weight = fix_weight
        assert not replace_all
        self.quantize = nn.Identity()

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AMMBottleneck):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, AMMBasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[AMMBasicBlock, AMMBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                amm_conv1x1(self.inplanes // self.subvec_len["1x1"], self.inplanes,
                            planes * block.expansion, stride, k=self.k,
                            temperature_config=self.temperature_config, fix_weight=self.fix_weight),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                k=self.k,
                subvec_len=self.subvec_len,
                temperature_config=self.temperature_config,
                fix_weight=self.fix_weight
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    k=self.k,
                    subvec_len=self.subvec_len,
                    temperature_config=self.temperature_config,
                    fix_weight=self.fix_weight
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.quantize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _amm_resnet(
    block: Type[Union[AMMBasicBlock, AMMBottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> AMMResNet:
    model = AMMResNet(block, layers, **kwargs)
    return model


def amm_resnet18_cifar(**kwargs: Any) -> AMMResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _amm_resnet(AMMBasicBlock, [2, 2, 2, 2], **kwargs)


def amm_resnet34_cifar(**kwargs: Any) -> AMMResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _amm_resnet(AMMBasicBlock, [3, 4, 6, 3], **kwargs)


def amm_resnet50_cifar(**kwargs: Any) -> AMMResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _amm_resnet(AMMBottleneck, [3, 4, 6, 3], **kwargs)


def amm_resnet101_cifar(**kwargs: Any) -> AMMResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _amm_resnet(AMMBottleneck, [3, 4, 23, 3], **kwargs)


def amm_resnet152_cifar(**kwargs: Any) -> AMMResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _amm_resnet(AMMBottleneck, [3, 8, 36, 3], **kwargs)
