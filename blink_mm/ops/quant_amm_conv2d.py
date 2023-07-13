import torch
import torch.nn as nn
import torch.nn.functional as F

from qat.ops import QuantizedTensor
from blink_mm.im2col import unfold
from .utils import calc_output_shape


class QuantizedAMMConv2dBatchNorm2dReLU(nn.Module):
    def __init__(
        self,
        ncodebooks,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        activation=None,
        k=16
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        assert self.activation in [None, "relu"]

        self.ncodebooks = ncodebooks
        d = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        assert d % self.ncodebooks == 0
        self.subvec_len = d // self.ncodebooks
        self.k = k

        self.register_quantized_tensor(
            "centroids",
            (self.ncodebooks, self.k, self.subvec_len),
            (self.ncodebooks, 1, 1)
        )
        # centroids
        self.register_quantized_tensor(
            "lut",
            (self.ncodebooks, self.k, self.out_channels)
        )
        # lut: zero point = 0
        self.register_quantized_tensor(
            "bias", (self.out_channels,), dtype=torch.int32
        )
        # bias: scale = lut.scale & zero point = 0

        self.register_scale_and_zero_point("output")

    @property
    def centroids(self):
        return QuantizedTensor(
            self.centroids_q,
            self.centroids_s,
            self.centroids_z
        )

    @property
    def lut(self):
        return QuantizedTensor(
            self.lut_q,
            self.lut_s,
            self.lut_z
        )

    @property
    def bias(self):
        return QuantizedTensor(
            self.bias_q,
            self.bias_s,
            self.bias_z
        )

    def register_scale_and_zero_point(self, name, shape=(1,), dtype=torch.int8):
        self.register_buffer(
            name + "_s",
            torch.empty(*shape).to(torch.float32)
        )
        self.register_buffer(
            name + "_z",
            torch.empty(*shape).to(dtype)
        )

    def register_quantized_tensor(self, name, q_shape, s_shape=(1,), dtype=torch.int8):
        self.register_buffer(
            name + "_q",
            torch.empty(*q_shape).to(dtype)
        )
        self.register_scale_and_zero_point(name, s_shape, dtype)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        elif self.activation is None:
            return x

    def extra_repr(self) -> str:
        return f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " \
            f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}"

    def _dist_argmin(self, x: QuantizedTensor):
        xy = torch.bmm(
            x.dequantize(),
            self.centroids.dequantize().permute(0, 2, 1)
        )
        y2 = (self.centroids.dequantize() ** 2).sum(dim=-1) \
            .unsqueeze(1)
        den = x.s * self.centroids.s
        dist = -2 * (xy / den).round() + (y2 / den).round()
        return torch.argmin(dist, dim=-1)

    def forward(self, x: QuantizedTensor) -> QuantizedTensor:
        b = x.shape[0]
        out_h, out_w = calc_output_shape(
            x.shape[2:], self.kernel_size, self.stride, self.padding)

        cols = unfold(x, self.kernel_size, self.stride, self.padding)
        x = cols.permute(0, 2, 1).map(lambda x: x.flatten(0, 1))
        x = x.reshape(x.shape[0], self.ncodebooks, self.subvec_len)
        x = x.permute(1, 0, 2)

        argmin = self._dist_argmin(x)
        one_hot = F.one_hot(argmin, num_classes=self.k).float()
        output = torch.bmm(one_hot, self.lut.dequantize()).sum(0)

        if self.bias is not None:
            output = output + self.bias.dequantize()
        output = output.reshape(b, out_h, out_w, self.out_channels)
        output = output.permute(0, 3, 1, 2)
        output = self._apply_activation(output)

        output_q = torch.clamp(
            output / self.output_s + self.output_z,
            torch.tensor(-128).to(output.device),
            torch.tensor(127).to(output.device)
        ).round().to(torch.int8)
        return QuantizedTensor(output_q, self.output_s, self.output_z)
