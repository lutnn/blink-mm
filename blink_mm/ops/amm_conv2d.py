from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from qat.ops import QuantizedTensor

from blink_mm.im2col import unfold
from .utils import calc_output_shape


class AMMConv2d(nn.Module):
    def __init__(
        self,
        ncodebooks,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        bias=True,
        temperature_config="inverse",
        k=16,
        fix_weight=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.ncodebooks = ncodebooks
        d = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        assert d % self.ncodebooks == 0
        self.subvec_len = d // self.ncodebooks
        self.k = k

        self.register_parameter(
            "centroids",
            nn.Parameter(torch.randn(self.ncodebooks, self.k, self.subvec_len))
        )
        self.register_parameter(
            "weight",
            nn.Parameter(torch.randn(
                self.ncodebooks, self.subvec_len, self.out_channels
            ))
        )
        assert temperature_config in ["manual", "direct", "inverse"]
        if temperature_config == "manual":
            self.register_buffer(
                "temperature",
                torch.tensor(1.0, dtype=torch.float32)
            )
        elif temperature_config == "direct":
            self.register_parameter(
                "temperature",
                nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            )
        elif temperature_config == "inverse":
            self.register_parameter(
                "inverse_temperature_logit",
                nn.Parameter(torch.randn(1))
            )
        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.randn(self.out_channels))
            )
        else:
            self.register_parameter('bias', None)

        if fix_weight:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " \
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}"

    @staticmethod
    def _quantize_lut(lut: torch.Tensor) -> QuantizedTensor:
        # Quantize weight to -128 ~ 127
        a = lut.min()
        b = lut.max()
        max_abs = torch.maximum(torch.abs(a), torch.abs(b))

        z = torch.zeros_like(a).to(torch.int8)
        s = max_abs / (127 - z.to(torch.float32))

        q = torch.clamp(
            lut / s + z,
            torch.tensor(-128).to(lut.device),
            torch.tensor(127).to(lut.device)
        ).round().to(torch.int8)
        return QuantizedTensor(q, s, z)

    def _forward(
        self, x: torch.Tensor, quantized_lut: QuantizedTensor
    ) -> Union[QuantizedTensor, torch.Tensor]:
        b = x.shape[0]
        out_h, out_w = calc_output_shape(
            x.shape[2:], self.kernel_size, self.stride, self.padding)

        cols = unfold(x, self.kernel_size, self.stride, self.padding)
        x = cols.permute(0, 2, 1).flatten(0, 1)
        x = x.reshape(x.shape[0], self.ncodebooks, self.subvec_len)
        x = x.permute(1, 0, 2)
        dist = torch.cdist(x, self.centroids)
        # (ncodebooks, bhw, k)
        if getattr(self, "temperature", None) is not None:
            multiplier = 1 / torch.max(
                self.temperature,
                torch.tensor(1e-6, device=self.temperature.device)
            )
        else:
            multiplier = F.softplus(self.inverse_temperature_logit) + 1
        attention = F.softmax(
            -dist * multiplier,
            dim=-1
        )
        # (ncodebooks, bhw, k)
        lut = torch.bmm(self.centroids, self.weight)
        # (ncodebooks, k, out_channels)
        real_output = torch.bmm(attention, lut).sum(0)

        one_hot = F.one_hot(dist.argmin(dim=-1), num_classes=self.k).float()
        quantized_output = torch.bmm(one_hot, lut).sum(0)
        output = real_output - (real_output - quantized_output).detach()
        # (bhw, out_channels)
        if self.bias is not None:
            output = output + self.bias
        output = output.reshape(b, out_h, out_w, self.out_channels)
        output = output.permute(0, 3, 1, 2)
        return output

    def forward(self, x):
        with torch.no_grad():
            fused_lut = torch.bmm(self.centroids, self.weight)
            quantized_lut = self._quantize_lut(fused_lut)

        return self._forward(x, quantized_lut)
