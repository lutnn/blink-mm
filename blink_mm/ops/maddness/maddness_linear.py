import torch
import torch.nn as nn
import torch.nn.functional as F

from .maddness_conv2d import MaddnessConv2d


class MaddnessLinear(nn.Module):
    def __init__(
        self,
        ncodebooks,
        in_features,
        out_features,
        bias=True
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ncodebooks = ncodebooks
        assert self.in_features % self.ncodebooks == 0
        self.subvec_len = self.in_features // self.ncodebooks
        self.k = 16

        self.register_buffer(
            "split_idxs",
            torch.randn(self.ncodebooks, 4).to(torch.long)
        )
        self.register_buffer(
            "split_vals",
            torch.randn(self.ncodebooks, 4, self.k // 2)
        )
        self.register_buffer(
            "lookup_tables",
            torch.randn(self.ncodebooks, self.k, self.out_features)
        )
        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.randn(self.out_features))
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        n, d = x.shape
        assert d == self.in_features

        output = torch.zeros(n, self.out_features).to(x.device)

        for i in range(self.ncodebooks):
            encoding = MaddnessConv2d._encode(
                x[:, self.subvec_len * i: self.subvec_len * (i + 1)],
                self.split_idxs[i],
                self.split_vals[i]
            )
            output += torch.matmul(
                F.one_hot(encoding.long(), num_classes=self.k).float(),
                self.lookup_tables[i]
            )

        # (bhw, out_channels)
        if self.bias is not None:
            output = output + self.bias

        return output
