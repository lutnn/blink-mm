import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .amm_conv2d import AMMConv2d


class AMMLinear(nn.Module):
    def __init__(
        self,
        ncodebooks,
        in_features,
        out_features,
        bias,
        k=16
    ):
        super().__init__()
        self.ncodebooks = ncodebooks
        self.in_features = in_features
        self.out_features = out_features
        assert self.in_features % self.ncodebooks == 0
        self.subvec_len = self.in_features // self.ncodebooks
        self.k = k

        self.register_parameter(
            "centroids",
            nn.Parameter(torch.randn(self.ncodebooks, self.k, self.subvec_len))
        )
        self.register_parameter(
            "weight",
            nn.Parameter(torch.randn(
                self.ncodebooks, self.subvec_len, self.out_features
            ))
        )
        self.register_parameter(
            "inverse_temperature_logit",
            nn.Parameter(torch.randn(1))
        )
        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.randn(self.out_features))
            )
        else:
            self.register_parameter('bias', None)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def _forward(self, x, quantized_lut):
        shape = x.shape[:-1]
        x = x.reshape(np.prod(shape), self.ncodebooks, self.subvec_len)
        x = x.permute(1, 0, 2)
        dist = torch.cdist(x, self.centroids)
        # (ncodebooks, b, k)
        attention = F.softmax(
            -dist * (F.softplus(self.inverse_temperature_logit) + 1),
            dim=-1
        )
        # (ncodebooks, b, k)
        lut = torch.bmm(self.centroids, self.weight)
        # (ncodebooks, k, out_features)
        real_output = torch.bmm(attention, lut).sum(0)

        one_hot = F.one_hot(dist.argmin(dim=-1), num_classes=self.k).float()
        quantized_output = torch.bmm(
            one_hot, quantized_lut.dequantize()).sum(0)
        output = real_output - (real_output - quantized_output).detach()
        # (b, out_features)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape, self.out_features)

    def forward(self, x):
        with torch.no_grad():
            fused_lut = torch.bmm(self.centroids, self.weight)
            quantized_lut = AMMConv2d._quantize_lut(fused_lut)

        return self._forward(x, quantized_lut)
