from tvm import relay
from tvm.relay.frontend.qnn_torch import _op, _expr, _get_numpy, infer_shape


def quantized_relu_impl(inputs, _):
    input_shape = infer_shape(inputs[0])
    input_scale = inputs[1]
    input_zero_point = inputs[2]
    output_scale = inputs[3]
    output_zero_point = inputs[4]

    requantize_input = _op.tensor.clip(
        inputs[0],
        _get_numpy(input_zero_point).item(),
        127
    )

    input_scale = _get_numpy(input_scale).tolist() * input_shape[0]
    input_zero_point = _get_numpy(input_zero_point).tolist() * input_shape[0]

    requantized = relay.qnn.op.requantize(
        requantize_input,
        relay.const(input_scale),
        relay.const(input_zero_point),
        _expr.const(_get_numpy(output_scale).item()),
        _expr.const(_get_numpy(output_zero_point).item()),
        out_dtype="int32",
        axis=0,
    )

    clip = _op.tensor.clip(requantized, -128, 127)
    return _op.cast(clip, dtype="int8")
