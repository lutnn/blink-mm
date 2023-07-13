from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm

from qat.ops import QuantizedConv2dBatchNorm2dReLU, QuantizedLinear, QuantizedOperator, QuantizedTensor
from qat.export.utils import fetch_module_by_name

from blink_mm.ops.quant_amm_conv2d import QuantizedAMMConv2dBatchNorm2dReLU
from blink_mm.transforms.handlers import _sync_centroids, TransferHandler
from blink_mm.transforms.utils import collect_input_tensors
from blink_mm.ops.amm_conv2d import AMMConv2d


def _find_scale_by_kl(arr, quantized_dtype="int8", num_bins=8001, num_quantized_bins=255):
    from blink_mm_native_lib import minimize_kl

    """https://github.com/apache/tvm/blob/main/python/tvm/relay/quantize/kl_divergence.py
    """
    assert isinstance(arr, np.ndarray)
    min_val = np.min(arr)
    max_val = np.max(arr)
    thres = max(abs(min_val), abs(max_val))

    if min_val >= 0 and quantized_dtype in ["uint8"]:
        # We need to move negative bins to positive bins to fit uint8 range.
        num_quantized_bins = num_quantized_bins * 2 + 1

    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-thres, thres))

    return minimize_kl(
        list(hist), list(hist_edges), num_bins, num_quantized_bins
    )


def _calc_conv_name(bn_name: str):
    splits = bn_name.split('.')
    if "bn" in splits[-1]:
        splits[-1] = splits[-1].replace("bn", "conv")
    else:
        splits[-1] = str(int(splits[-1]) - 1)
    return '.'.join(splits)


def _collect_quant_stats(model, calibrate_dataset, is_bn2d, activation):
    module_names = OrderedDict([
        (module, name) for name, module in model.named_modules()
    ])

    dic = {
        "min": {},
        "max": {},
        "s": {},
        "z": {},
    }

    if not dist.is_initialized() or dist.get_rank() == 0:
        output_tensors = collect_input_tensors(
            model, calibrate_dataset, is_bn2d,
            is_input_tensor=False
        )

        for bn2d, output_tensor in tqdm(list(output_tensors.items()), desc="computing quantization statistics"):
            with torch.no_grad():
                if activation(bn2d) == "relu":
                    output_tensor = F.relu(output_tensor)
            scale = _find_scale_by_kl(output_tensor.detach().cpu().numpy())
            if output_tensor.min() >= 0:
                max = scale
                min = 0
                z = np.round(127 - max * 255 / (max - min)).astype(np.int8)
            else:
                max = scale
                min = -scale
                z = np.array(0)
            s = max / (127 - z.astype(np.float32))
            dic["min"][bn2d] = torch.tensor(min)
            dic["max"][bn2d] = torch.tensor(max)
            dic["s"][bn2d] = torch.tensor(s)
            dic["z"][bn2d] = torch.tensor(z)

    modules = list(filter(is_bn2d, module_names.keys()))
    _sync_centroids(modules, dic["min"])
    _sync_centroids(modules, dic["max"])
    _sync_centroids(modules, dic["s"])
    _sync_centroids(modules, dic["z"])
    return dic


class QuantizedOperatorTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset):
        super().__init__(model, target_model, calibrate_dataset)

        self.dic = self._calc_running_min_and_max(
            model, target_model, calibrate_dataset)

    def transfer(self, module, target):
        target.num_batches_tracked.data.copy_(torch.tensor(1))
        target.running_min.data.copy_(self.dic["min"][module])
        target.running_max.data.copy_(self.dic["max"][module])

    @staticmethod
    def _calc_running_min_and_max(model, target_model, calibrate_dataset):
        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        def is_quant_op(m):
            try:
                target_module = fetch_module_by_name(
                    target_model, module_names[m])
                return isinstance(target_module, QuantizedOperator)
            except:
                return False

        def activation(m):
            return None

        return _collect_quant_stats(model, calibrate_dataset, is_quant_op, activation)


class QuantizedConv2dBatchNorm2dReLUTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset):
        super().__init__(model, target_model, calibrate_dataset)

        self.dic = self._calc_running_min_and_max(
            model, target_model, calibrate_dataset)

    def transfer_conv2d_bn2d(
        self, conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d,
        target: QuantizedConv2dBatchNorm2dReLU
    ):
        self.transfer_conv2d(conv2d, target.conv2d)
        self.transfer_bn2d(bn2d, target.bn2d)
        target.num_batches_tracked.data.copy_(torch.tensor(1))
        target.running_min.data.copy_(self.dic["min"][bn2d])
        target.running_max.data.copy_(self.dic["max"][bn2d])

    @staticmethod
    def _calc_running_min_and_max(model, target_model, calibrate_dataset):
        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        def is_bn2d(m):
            if not isinstance(m, nn.BatchNorm2d):
                return False
            target_module = fetch_module_by_name(
                target_model, _calc_conv_name(module_names[m]))
            return isinstance(target_module, QuantizedConv2dBatchNorm2dReLU)

        def activation(m):
            target_module = fetch_module_by_name(
                target_model, _calc_conv_name(module_names[m]))
            return getattr(target_module, "activation")

        return _collect_quant_stats(model, calibrate_dataset, is_bn2d, activation)


class QuantizedLinearTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset):
        super().__init__(model, target_model, calibrate_dataset)

        self.dic = self._calc_running_min_and_max(
            model, target_model, calibrate_dataset)

    def transfer(self, linear: nn.Linear, target: QuantizedLinear):
        self.transfer_linear(linear, target.linear)
        target.num_batches_tracked.data.copy_(torch.tensor(1))
        target.running_min.data.copy_(self.dic["min"][linear])
        target.running_max.data.copy_(self.dic["max"][linear])

    @staticmethod
    def _calc_running_min_and_max(model, target_model, calibrate_dataset):
        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        def is_linear(m):
            if not isinstance(m, nn.Linear):
                return False
            target_module = fetch_module_by_name(target_model, module_names[m])
            return isinstance(target_module, QuantizedLinear)

        def activation(m):
            return None

        return _collect_quant_stats(model, calibrate_dataset, is_linear, activation)


class QuantizedAMMConv2dBatchNorm2dReLUTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset):
        super().__init__(model, target_model, calibrate_dataset)

        self.dic = self._calc_output_scale_and_zero_point(
            model, target_model, calibrate_dataset)

    @staticmethod
    def _calc_output_scale_and_zero_point(model, target_model, calibrate_dataset):
        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        def is_bn2d(m):
            if not isinstance(m, nn.BatchNorm2d):
                return False
            target_module = fetch_module_by_name(
                target_model, _calc_conv_name(module_names[m]))
            return isinstance(target_module, QuantizedAMMConv2dBatchNorm2dReLU)

        def activation(m):
            target_module = fetch_module_by_name(
                target_model, _calc_conv_name(module_names[m]))
            return getattr(target_module, "activation")

        return _collect_quant_stats(model, calibrate_dataset, is_bn2d, activation)

    @staticmethod
    def _quantize_centroids(tensor, quantized_tensor):
        # (ncodebooks, k, subvec_len)
        max, _ = tensor.flatten(1).max(dim=1)
        min, _ = tensor.flatten(1).min(dim=1)
        z = (127 - max * 255 / (max - min)).round().to(torch.int8)
        s = max / (127 - z.to(torch.float32))
        z = z.reshape(-1, 1, 1)
        s = s.reshape(-1, 1, 1)
        q = torch.clamp(
            tensor / s + z,
            torch.tensor(-128).to(tensor.device),
            torch.tensor(127).to(tensor.device)
        ).round().to(torch.int8)
        quantized_tensor.q.data.copy_(q)
        quantized_tensor.s.data.copy_(s)
        quantized_tensor.z.data.copy_(z)

    @staticmethod
    def _quantize_lut(tensor, quantized_tensor):
        a = tensor.min()
        b = tensor.max()
        max_abs = torch.maximum(torch.abs(a), torch.abs(b))
        z = torch.zeros_like(a).to(torch.int8)
        s = max_abs / (127 - z.to(torch.float32))
        q = torch.clamp(
            tensor / s + z,
            torch.tensor(-128).to(tensor.device),
            torch.tensor(127).to(tensor.device)
        ).round().to(torch.int8)
        quantized_tensor.q.data.copy_(q)
        quantized_tensor.s.data.copy_(s)
        quantized_tensor.z.data.copy_(z)

    @staticmethod
    def _get_fused_lut_and_bias(conv2d: AMMConv2d, bn2d: nn.BatchNorm2d):
        sqrt_var = torch.sqrt(bn2d.running_var + bn2d.eps)
        lut = torch.bmm(conv2d.centroids, conv2d.weight)
        quantized_lut = QuantizedTensor(
            torch.empty_like(lut).to(torch.int8),
            torch.empty(1).to(lut.device, torch.float32),
            torch.empty(1).to(lut.device, torch.int8)
        )
        cls = QuantizedAMMConv2dBatchNorm2dReLUTransferHandler
        cls._quantize_lut(lut, quantized_lut)
        # (ncodebooks, k, out_channels)
        fused_lut = quantized_lut.dequantize() * bn2d.weight / sqrt_var
        bias = torch.zeros_like(bn2d.running_mean) \
            if conv2d.bias is None else conv2d.bias
        fused_bias = (bias - bn2d.running_mean) * \
            bn2d.weight / sqrt_var + bn2d.bias
        return fused_lut, fused_bias

    @staticmethod
    def _quantize_bias(tensor, quantized_tensor):
        # quantized_tensor.s already assigned
        z = torch.zeros_like(quantized_tensor.s).to(torch.int32)
        q = (tensor / quantized_tensor.s.to(tensor.device)).round().to(torch.int32)
        quantized_tensor.z.data.copy_(z)
        quantized_tensor.q.data.copy_(q)

    def transfer_conv2d_bn2d(
        self, conv2d: AMMConv2d, bn2d: nn.BatchNorm2d,
        target: QuantizedAMMConv2dBatchNorm2dReLU
    ):
        self._quantize_centroids(conv2d.centroids, target.centroids)
        fused_lut, fused_bias = self._get_fused_lut_and_bias(conv2d, bn2d)
        self._quantize_lut(fused_lut, target.lut)
        target.bias_s.data.copy_(target.lut_s)
        self._quantize_bias(fused_bias, target.bias)
        target.output_s.data.copy_(self.dic["s"][bn2d])
        target.output_z.data.copy_(self.dic["z"][bn2d])
