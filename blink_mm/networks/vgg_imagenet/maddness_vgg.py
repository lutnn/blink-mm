from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

from blink_mm.ops.maddness.maddness_conv2d import MaddnessConv2d

__all__ = [
    "maddness_vgg11",
    "maddness_vgg11_bn",
]


class MaddnessVGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(
    cfg: List[Union[str, int]], batch_norm: bool = False, subvec_len=9,
    replace_all=False
) -> nn.Sequential:
    assert not replace_all

    first = True

    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            if first:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                first = False
            else:
                ncodebooks = in_channels * 9 // subvec_len
                conv2d = MaddnessConv2d(
                    ncodebooks, in_channels, v,
                    (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _maddness_vgg(
    cfg: str, batch_norm: bool, subvec_len=9,
    replace_all=False, **kwargs: Any
) -> MaddnessVGG:
    return MaddnessVGG(make_layers(
        cfgs[cfg], batch_norm=batch_norm, subvec_len=subvec_len,
        replace_all=replace_all
    ), **kwargs)


def maddness_vgg11(**kwargs: Any) -> MaddnessVGG:
    return _maddness_vgg("A", False, **kwargs)


def maddness_vgg11_bn(**kwargs: Any) -> MaddnessVGG:
    return _maddness_vgg("A", True, **kwargs)
