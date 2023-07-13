from collections import OrderedDict

import torch

from qat.export.handlers import Handler, _WrapperModule
from blink_mm.ops.amm_conv2d import AMMConv2d

from blink_mm.transforms.export.functions import AMMConv2dFn, AMMLinearFn, QuantizedAMMConv2dBatchNorm2dReLUFn


class AMMConv2dHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        bias = torch.zeros(
            module.out_channels) if module.bias is None else module.bias
        fused_lut = torch.bmm(module.centroids, module.weight)
        quantized_lut = module._quantize_lut(fused_lut)
        self.args[module] = OrderedDict({
            "bias": bias.detach(),
            "centroids": module.centroids.detach(),
            "lut": quantized_lut.q,
            "s": quantized_lut.s.detach(),
            "output_shape": torch.tensor(outputs.shape),
            "kernel_size": torch.tensor(module.kernel_size),
            "stride": torch.tensor(module.stride),
            "padding": torch.tensor(module.padding),
        })

    def replace_module(self, module):
        return _WrapperModule(lambda x: AMMConv2dFn.apply(x, *self.args[module].values()))


class AMMLinearHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        bias = torch.zeros(
            module.out_features) if module.bias is None else module.bias
        fused_lut = torch.bmm(module.centroids, module.weight)
        quantized_lut = AMMConv2d._quantize_lut(fused_lut)
        self.args[module] = OrderedDict({
            "bias": bias.detach(),
            "centroids": module.centroids.detach(),
            "lut": quantized_lut.q,
            "s": quantized_lut.s.detach(),
            "output_shape": torch.tensor(outputs.shape)
        })

    def replace_module(self, module):
        return _WrapperModule(lambda x: AMMLinearFn.apply(x, *self.args[module].values()))


class QuantizedAMMConv2dBatchNorm2dReLUHandler(Handler):
    def forward_hook(self, module, inputs, outputs):
        self.args[module] = OrderedDict({
            "input_scale": inputs[0].s.detach(),
            "input_zero_point": inputs[0].z,
            "int_centroids": module.centroids.q,
            "centroids_scale": module.centroids.s.detach().flatten(),
            "centroids_zero_point": module.centroids.z.flatten(),
            "int_lut": module.lut.q,
            "lut_scale": module.lut.s.detach(),
            "int_bias": module.bias.q,
            "output_scale": module.output_s.detach(),
            "output_zero_point": module.output_z,
            "output_dtype": torch.int8,
            "out_shape": torch.tensor(outputs.shape),
            "kernel_size": torch.tensor(module.kernel_size),
            "padding": torch.tensor(module.padding),
            "stride": torch.tensor(module.stride),
        })

    def replace_module(self, module):
        return _WrapperModule(
            lambda x: QuantizedAMMConv2dBatchNorm2dReLUFn.apply(
                x, *self.args[module].values()
            ))
