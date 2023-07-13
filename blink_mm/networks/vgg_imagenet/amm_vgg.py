from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn

from blink_mm.ops.amm_conv2d import AMMConv2d

__all__ = [
    "amm_vgg11",
    "amm_vgg11_bn",
]


class AMMVGG(nn.Module):
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
    cfg: List[Union[str, int]], batch_norm: bool = False, k=16, subvec_len=9,
    temperature_config="inverse", fix_weight=False, replace_all=False
) -> nn.Sequential:
    assert temperature_config == "inverse"
    assert not fix_weight
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
                conv2d = AMMConv2d(
                    ncodebooks, in_channels, v,
                    (3, 3), (1, 1), (1, 1), k=k)
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


def _amm_vgg(
    cfg: str, batch_norm: bool, k=16, subvec_len=9,
    temperature_config="inverse", fix_weight=False, replace_all=False, **kwargs: Any
) -> AMMVGG:
    return AMMVGG(make_layers(
        cfgs[cfg], batch_norm=batch_norm, k=k, subvec_len=subvec_len,
        temperature_config=temperature_config, fix_weight=fix_weight, replace_all=replace_all
    ), **kwargs)


def amm_vgg11(**kwargs: Any) -> AMMVGG:
    return _amm_vgg("A", False, **kwargs)


def amm_vgg11_bn(**kwargs: Any) -> AMMVGG:
    return _amm_vgg("A", True, **kwargs)
