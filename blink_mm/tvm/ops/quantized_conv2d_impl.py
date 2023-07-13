from tvm import relay
from tvm.relay.frontend.qnn_torch import _op, _expr, _get_numpy, _get_scalar, infer_shape

# copied from qnn_torch, except that uint8 is changed to int8


def _do_bias_and_requantize(
    output, bias, input_scale, weight_scale, output_scale, output_zero_point, with_relu
):
    """Output processing for conv and linear"""
    # this is a vector for per channel case
    requant_input_scale = _expr.const(
        _get_numpy(input_scale) * _get_numpy(weight_scale))
    # Torch does bias add and requanize scale in fp32
    # refer to third_party/fbgemm/include/fbgemm/OutputProcessing-inl.h
    # Instead, we do bias add in int32 and use qnn requantize, which needs
    # integer input.
    # We observed no loss in accuracy in doing this way, and it is better
    # for tvm because bias quantization can be done at compile time
    # Instead, the torch way requires rounding of activation at runtime

    if bias is not None:
        requantize_input = _op.nn.bias_add(output, bias)
    else:
        requantize_input = output

    requantized = relay.qnn.op.requantize(
        requantize_input,
        requant_input_scale,
        relay.const(0, "int32"),
        output_scale,
        output_zero_point,
        out_dtype="int32",
        axis=1,
    )
    clip_min = -128
    if with_relu:
        clip_min = _get_scalar(output_zero_point)

    clip = _op.tensor.clip(requantized, clip_min, 127)
    return _op.cast(clip, dtype="int8")


def _quantized_conv2d(with_relu=False):
    def _impl(inputs, _):
        # refer to src/ATen/native/quantized/cpu/qconv.cpp
        # inputs[0]: input tensor
        # inputs[1]: (weight, scale, zero_point, bias)
        # inputs[2-5]: stride, padding, dilation, groups
        # inputs[6]: output_scale
        # inputs[7]: output_zero_point
        # inputs[8]: input_scale (added manually by frontend)
        # inputs[9]: input_zero_point (added manually by frontend)
        conv_params = inputs[1]
        weight = conv_params[0]
        weight_scale = conv_params[1]
        weight_zero_point = conv_params[2]
        bias = conv_params[3]

        if len(conv_params) > 4:
            # Torch 1.6 or newer case
            strides = conv_params[4]
            padding = conv_params[5]
            dilation = conv_params[6]
            groups = conv_params[7]

            output_scale = _expr.const(inputs[2])
            output_zero_point = _expr.const(inputs[3])

            assert len(inputs) == 6, "Input quant params not found in op inputs"

            # These are manually added by add_input_quant_params_to_op_inputs above
            # In torch, they are retrieved from QTensor data structure at runtime
            input_scale = _expr.const(inputs[4])
            input_zero_point = _expr.const(inputs[5])
        else:
            strides = inputs[2]
            padding = inputs[3]
            dilation = inputs[4]
            groups = inputs[5]
            output_scale = _expr.const(inputs[6])
            output_zero_point = _expr.const(inputs[7])

            assert len(inputs) == 10, \
                "Input quant params not found in op inputs"

            input_scale = _expr.const(inputs[8])
            input_zero_point = _expr.const(inputs[9])

        weight_shape = infer_shape(weight)
        kernel_size = (weight_shape[2], weight_shape[3])
        out_channels = weight_shape[0]

        if padding[0] != 0 or padding[1] != 0:
            pad_val = _get_scalar(input_zero_point)
            inp = _op.nn.pad(
                inputs[0],
                pad_width=((0, 0), (0, 0),
                           (padding[0], padding[0]), (padding[1], padding[1])),
                pad_value=float(pad_val),
            )
        else:
            inp = inputs[0]

        # padding is (0, 0) because we did explicit pad op with
        # pad value being zero point above
        conv_out = relay.qnn.op.conv2d(
            inp,
            weight,
            input_zero_point,
            weight_zero_point,
            input_scale,
            weight_scale,
            kernel_size=kernel_size,
            dilation=dilation,
            strides=strides,
            padding=(0, 0),
            groups=groups,
            channels=out_channels,
        )

        return _do_bias_and_requantize(
            conv_out, bias, input_scale, weight_scale, output_scale, output_zero_point, with_relu
        )

    return _impl


def quantized_conv2d_impl(inputs, input_types):
    func = _quantized_conv2d(False)

    stride = _get_numpy(inputs[12]).tolist()
    padding = _get_numpy(inputs[11]).tolist()
    padding = [padding[0], padding[2]]
    dilation = _get_numpy(inputs[14]).tolist()
    groups = inputs[13]

    return func(
        [
            inputs[0],  # 0
            (
                inputs[3], inputs[4],
                relay.cast(inputs[5], "int32"),
                inputs[8]
            ),  # 1
            stride,  # 2
            padding,  # 3
            dilation,  # 4
            groups,  # 5
            _get_numpy(inputs[6]).item(),  # 6
            _get_numpy(inputs[7]).item(),  # 7
            _get_numpy(inputs[1]).item(),  # 8
            _get_numpy(inputs[2]).item(),  # 9
        ],
        None
    )
