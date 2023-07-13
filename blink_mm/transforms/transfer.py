from collections import OrderedDict

from torch.utils.data import Dataset
import torch.nn as nn

from qat.export.utils import fetch_module_by_name
from qat.ops import *

from blink_mm.ops.quant_amm_conv2d import QuantizedAMMConv2dBatchNorm2dReLU
from blink_mm.ops.amm_conv2d import AMMConv2d
from blink_mm.ops.amm_linear import AMMLinear
from .handlers import *
from .quantize.handlers import *


QUANTIZE_PASS = OrderedDict([
    (QuantizedConv2dBatchNorm2dReLU, QuantizedConv2dBatchNorm2dReLUTransferHandler),
    (QuantizedAMMConv2dBatchNorm2dReLU,
     QuantizedAMMConv2dBatchNorm2dReLUTransferHandler),
    (QuantizedLinear, QuantizedLinearTransferHandler),
    (Quantize, QuantizedOperatorTransferHandler),
    (QuantizedAdd, QuantizedOperatorTransferHandler),
    (QuantizedReLU, QuantizedOperatorTransferHandler),
    (QuantizedAdaptiveAvgPool2d, QuantizedOperatorTransferHandler)
])

AMM_PASS = OrderedDict([
    (AMMConv2d, AMMConv2dTransferHandler),
    (AMMLinear, AMMLinearTransferHandler),
    (nn.BatchNorm2d, TransferHandler),
    (nn.Linear, TransferHandler),
    (nn.Conv2d, TransferHandler),
])

MADDNESS_PASS = OrderedDict([
    (MaddnessLinear, MaddnessLinearTransferHandler),
    (MaddnessConv2d, MaddnessConv2dTransferHandler),
    (nn.BatchNorm2d, TransferHandler),
    (nn.Linear, TransferHandler),
    (nn.Conv2d, TransferHandler),
])


def transfer(
    model, target_model, calibrate_dataset: Dataset,
    pass_type: dict
):
    handler_instances = {
        handler_type: handler_type(model, target_model, calibrate_dataset)
        for handler_type in OrderedDict.fromkeys(pass_type.values())
    }
    handlers = {
        key: handler_instances[value]
        for key, value in pass_type.items()
    }

    def _calc_bn_name(conv_name: str):
        splits = conv_name.split('.')
        if "conv" in splits[-1]:
            # e.g. conv1 -> bn1
            splits[-1] = splits[-1].replace("conv", "bn")
        else:
            try:
                # e.g. layers.0 (Conv2d) -> layers.1 (BatchNorm2d)
                splits[-1] = str(int(splits[-1]) + 1)
            except:
                # e.g. a single convolution
                return None
        return '.'.join(splits)

    def _choose_transfer_func(model, module, target, name):
        if "Conv" in module.__class__.__name__:
            bn_name = _calc_bn_name(name)
            if bn_name is not None:
                try:
                    bn2d = fetch_module_by_name(model, bn_name)
                except:
                    return "transfer", [module, target]
                return "transfer_conv2d_bn2d", [module, bn2d, target]
            else:
                return "transfer", [module, target]
        else:
            return "transfer", [module, target]

    for name, module in model.named_modules():
        try:
            target = fetch_module_by_name(target_model, name)
        except:
            continue
        handler = handlers.get(type(target), None)
        if handler is None:
            continue
        func, args = _choose_transfer_func(model, module, target, name)
        getattr(handler, func)(*args)
