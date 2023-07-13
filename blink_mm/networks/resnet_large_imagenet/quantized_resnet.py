from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from qat.ops import \
    QuantizedAdd, QuantizedReLU, QuantizedConv2dBatchNorm2dReLU, \
    QuantizedMaxPool2d, QuantizedAdaptiveAvgPool2d, QuantizedLinear, \
    Quantize, QuantizedFlatten, QuantizedTensor

from blink_mm.ops.quant_amm_conv2d import QuantizedAMMConv2dBatchNorm2dReLU


__all__ = ['quantized_resnet18', 'quantized_resnet34',
           'quantized_resnet50', 'quantized_resnet101', 'quantized_resnet152']


def quantized_conv3x3(ncodebooks, in_planes: int, out_planes: int, stride: int = 1, activation=None):
    return QuantizedAMMConv2dBatchNorm2dReLU(
        ncodebooks,
        in_planes,
        out_planes,
        (3, 3),
        (stride, stride),
        (1, 1),
        activation
    )


def quantized_conv1x1(ncodebooks, in_planes: int, out_planes: int, stride: int = 1, activation=None):
    return QuantizedAMMConv2dBatchNorm2dReLU(
        ncodebooks, in_planes, out_planes, (1, 1), (stride, stride), activation=activation
    )


class QuantizedBasicBlock(nn.Module):
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
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = quantized_conv3x3(
            inplanes, inplanes, planes, stride, "relu")
        self.conv2 = quantized_conv3x3(planes, planes, planes)
        self.add = QuantizedAdd()
        self.relu = QuantizedReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class QuantizedBottleneck(nn.Module):
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
    ) -> None:
        super().__init__()
        assert dilation == 1
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = quantized_conv1x1(
            inplanes // 4, inplanes, width, activation="relu")
        self.conv2 = quantized_conv3x3(
            width, width, width, stride, activation="relu")
        self.conv3 = quantized_conv1x1(
            width // 4, width, planes * self.expansion)
        self.add = QuantizedAdd()
        self.relu = QuantizedReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class QuantizedResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[QuantizedBasicBlock, QuantizedBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super().__init__()

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
        self.quantize = Quantize()
        self.conv1 = QuantizedConv2dBatchNorm2dReLU(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, activation="relu"
        )
        self.maxpool = QuantizedMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = QuantizedAdaptiveAvgPool2d((1, 1))
        self.fc = QuantizedLinear(512 * block.expansion, num_classes)
        self.flatten = QuantizedFlatten(1)

    def _make_layer(
        self,
        block: Type[Union[QuantizedBasicBlock, QuantizedBottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                quantized_conv1x1(self.inplanes // 4, self.inplanes,
                                  planes * block.expansion, stride),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation
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
                )
            )

        return nn.Sequential(*layers)

    def dequantize(self, x):
        if isinstance(x, QuantizedTensor):
            return x.dequantize()
        else:
            return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.quantize(x)
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return self.dequantize(x)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _quantized_resnet(
    block: Type[Union[QuantizedBasicBlock, QuantizedBottleneck]],
    layers: List[int],
    **kwargs: Any,
) -> QuantizedResNet:
    return QuantizedResNet(block, layers, **kwargs)


def quantized_resnet18(**kwargs: Any) -> QuantizedResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _quantized_resnet(QuantizedBasicBlock, [2, 2, 2, 2], **kwargs)


def quantized_resnet34(**kwargs: Any) -> QuantizedResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _quantized_resnet(QuantizedBasicBlock, [3, 4, 6, 3], **kwargs)


def quantized_resnet50(**kwargs: Any) -> QuantizedResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _quantized_resnet(QuantizedBottleneck, [3, 4, 6, 3], **kwargs)


def quantized_resnet101(**kwargs: Any) -> QuantizedResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _quantized_resnet(QuantizedBottleneck, [3, 4, 23, 3], **kwargs)


def quantized_resnet152(**kwargs: Any) -> QuantizedResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _quantized_resnet(QuantizedBottleneck, [3, 8, 36, 3], **kwargs)
