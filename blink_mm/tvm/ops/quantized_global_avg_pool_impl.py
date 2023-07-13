from tvm import relay
from tvm.relay.frontend import qnn_torch
from tvm.relay.frontend.pytorch import _op
from tvm.relay.frontend.qnn_torch import _get_numpy, infer_shape, _expr


def adaptive_avg_pool(op, inputs, _):
    data = inputs[0]
    output_size = inputs[1]

    def func(x):
        return op(x, output_size=output_size)

    return qnn_torch.apply_with_upcast(data, func)


def quantized_global_avg_pool_impl(inputs, input_types):
    input_shape = infer_shape(inputs[0])
    input_scale = inputs[1]
    input_zero_point = inputs[2]
    output_scale = inputs[3]
    output_zero_point = inputs[4]

    requantize_input = adaptive_avg_pool(
        _op.nn.adaptive_avg_pool2d,
        [inputs[0], (1, 1)],
        None
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
