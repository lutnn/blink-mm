import sys
import os.path as osp

import numpy as np

from qat.ops import QuantizedConv2dBatchNorm2dReLU, QuantizedLinear

from blink_mm.ops.amm_conv2d import AMMConv2d
from blink_mm.ops.quant_amm_conv2d import QuantizedAMMConv2dBatchNorm2dReLU
from blink_mm.ops.maddness.maddness_conv2d import MaddnessConv2d
from blink_mm.ops.utils import calc_output_shape


class AMMConv2dForwardHook:
    module = AMMConv2d

    def __call__(self, dst):
        def func(module, input_tensors, output_tensors):
            assert isinstance(module, AMMConv2d)
            out_h, out_w = calc_output_shape(
                input_tensors[0].shape[2:], module.kernel_size, module.stride, module.padding)
            b, in_channels, _, _ = input_tensors[0].shape
            n = b * out_h * out_w
            d = in_channels * module.kernel_size[0] * module.kernel_size[1]
            m = module.out_channels
            muladds = n * d * module.k
            adds = n * module.ncodebooks * m
            mem = 4 * np.prod(input_tensors[0].shape) + \
                4 * module.k * d + \
                1 * module.ncodebooks * module.k * m + 4 * n * m
            params = 4 * module.k * d + 1 * module.ncodebooks * module.k * m

            dst["muladds"] = dst.get("muladds", 0) + muladds
            dst["adds"] = dst.get("adds", 0) + adds
            dst["mem"] = dst.get("mem", 0) + mem
            dst["max_mem"] = max(dst.get("max_mem", 0), mem)
            dst["params"] = dst.get("params", 0) + params
            dst["dt"] = dst.get("dt", 0) + n * module.ncodebooks * 12

        return func


class QuantizedAMMConv2dBatchNorm2dReLUForwardHook:
    module = QuantizedAMMConv2dBatchNorm2dReLU

    def __call__(self, dst):
        def func(module, input_tensors, output_tensors):
            assert isinstance(module, QuantizedAMMConv2dBatchNorm2dReLU)
            out_h, out_w = calc_output_shape(
                input_tensors[0].shape[2:], module.kernel_size, module.stride, module.padding)
            b, in_channels, _, _ = input_tensors[0].shape
            n = b * out_h * out_w
            d = in_channels * module.kernel_size[0] * module.kernel_size[1]
            m = module.out_channels
            muladds = n * d * module.k
            adds = n * module.ncodebooks * m
            mem = 1 * np.prod(input_tensors[0].shape) + \
                1 * module.k * d + \
                1 * module.ncodebooks * module.k * m + 1 * n * m
            params = 1 * module.k * d + 1 * module.ncodebooks * module.k * m

            dst["muladds"] = dst.get("muladds", 0) + muladds
            dst["adds"] = dst.get("adds", 0) + adds
            dst["mem"] = dst.get("mem", 0) + mem
            dst["max_mem"] = max(dst.get("max_mem", 0), mem)
            dst["params"] = dst.get("params", 0) + params
            dst["dt"] = dst.get("dt", 0) + n * module.ncodebooks * 12

        return func


class QuantizedConv2dBatchNorm2dReLUForwardHook:
    module = QuantizedConv2dBatchNorm2dReLU

    def __call__(self, dst):
        def func(module, input_tensors, output_tensors):
            assert isinstance(module, QuantizedConv2dBatchNorm2dReLU)
            b, in_channels, h, w = input_tensors[0].shape
            out_h, out_w = calc_output_shape(
                input_tensors[0].shape[2:],
                module.conv2d.kernel_size,
                module.conv2d.stride,
                module.conv2d.padding
            )
            muladds = b * module.conv2d.out_channels * out_h * out_w * \
                module.conv2d.kernel_size[0] * \
                module.conv2d.kernel_size[1] * module.conv2d.in_channels
            mem = b * module.conv2d.in_channels * h * w + np.prod(module.conv2d.weight.shape) + \
                b * module.conv2d.out_channels * out_h * out_w
            params = np.prod(module.conv2d.weight.shape)

            dst["muladds"] = dst.get("muladds", 0) + muladds
            dst["mem"] = dst.get("mem", 0) + mem
            dst["max_mem"] = max(dst.get("max_mem", 0), mem)
            dst["params"] = dst.get("params", 0) + params

        return func


class QuantizedLinearForwardHook:
    module = QuantizedLinear

    def __call__(self, dst):
        def func(module, input_tensors, output_tensors):
            assert isinstance(module, QuantizedLinear)
            in_features, out_features = module.linear.in_features, module.linear.out_features
            b, _ = input_tensors[0].shape
            muladds = b * in_features * out_features
            mem = b * in_features + b * out_features + \
                in_features * out_features
            params = in_features * out_features

            dst["muladds"] = dst.get("muladds", 0) + muladds
            dst["mem"] = dst.get("mem", 0) + mem
            dst["max_mem"] = max(dst.get("max_mem", 0), mem)
            dst["params"] = dst.get("params", 0) + params

        return func


class MaddnessConv2DForwardHook:
    module = MaddnessConv2d

    def __call__(self, dst):
        def func(module, input_tensors, output_tensors):
            assert isinstance(module, MaddnessConv2d)
            b, in_channels, h, w = input_tensors[0].shape
            out_h, out_w = calc_output_shape(
                input_tensors[0].shape[2:],
                module.kernel_size,
                module.stride,
                module.padding
            )
            n = b * out_h * out_w
            d = in_channels * module.kernel_size[0] * module.kernel_size[1]
            m = module.out_channels
            mem = 4 * np.prod(input_tensors[0].shape) + \
                4 * module.ncodebooks * 36 + 2 * 1 * module.ncodebooks * n + \
                4 * module.ncodebooks * module.k * m + 4 * n * m
            adds = n * module.ncodebooks * m
            params = 4 * module.ncodebooks * 36 + 2 * 1 * module.ncodebooks * n + \
                4 * module.ncodebooks * module.k * m

            dst["adds"] = dst.get("adds", 0) + adds
            dst["mem"] = dst.get("mem", 0) + mem
            dst["max_mem"] = max(dst.get("max_mem", 0), mem)
            dst["params"] = dst.get("params", 0) + params

        return func
