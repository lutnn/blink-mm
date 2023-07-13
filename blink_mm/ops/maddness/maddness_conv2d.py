import torch
import torch.nn as nn
import torch.nn.functional as F

from blink_mm.im2col import unfold
from blink_mm.ops.utils import calc_output_shape


class MaddnessConv2d(nn.Module):
    def __init__(
        self,
        ncodebooks,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1),
        padding=(0, 0),
        bias=True
    ) -> None:
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
            torch.randn(self.ncodebooks, self.k, self.out_channels)
        )
        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.randn(self.out_channels))
            )
        else:
            self.register_parameter('bias', None)

    @staticmethod
    def _encode(subspace, split_idxs, split_vals):
        n, _ = subspace.shape

        buckets = [torch.arange(n).to(subspace.device)]
        for t in range(4):
            new_buckets = []
            for bucket, split_val in zip(buckets, split_vals[t]):
                new_buckets.append(
                    bucket[subspace[bucket, split_idxs[t]] < split_val])
                new_buckets.append(
                    bucket[subspace[bucket, split_idxs[t]] >= split_val])
            buckets = new_buckets

        encoding = torch.zeros(n).to(torch.long).to(subspace.device)
        for i, bucket in enumerate(buckets):
            encoding[bucket] = i
        return encoding

    def forward(self, x):
        b = x.shape[0]
        out_h, out_w = calc_output_shape(
            x.shape[2:], self.kernel_size, self.stride, self.padding)

        cols = unfold(x, self.kernel_size, self.stride, self.padding)
        x = cols.permute(0, 2, 1).flatten(0, 1)

        n, d = x.shape
        assert d == self.ncodebooks * self.subvec_len

        output = torch.zeros(n, self.out_channels).to(x.device)

        for i in range(self.ncodebooks):
            encoding = self._encode(
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
        output = output.reshape(b, out_h, out_w, self.out_channels)
        output = output.permute(0, 3, 1, 2)

        return output
