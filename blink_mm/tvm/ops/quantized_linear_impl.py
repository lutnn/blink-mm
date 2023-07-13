from tvm import relay
from tvm.relay.frontend.qnn_torch import _get_numpy, infer_shape, _expr

from blink_mm.tvm.ops.quantized_conv2d_impl import _do_bias_and_requantize


def _linear(with_relu=False):
    # similar to conv
    def _impl(inputs, _):
        weight = inputs[1][0]
        weight_scale = inputs[1][1]
        weight_zero_point = inputs[1][2]
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        assert len(inputs) == 6, "Input quant params not found in op inputs"
        # Manually added by add_input_quant_params_to_op_inputs above
        input_scale = _expr.const(inputs[4])
        input_zero_point = _expr.const(inputs[5])

        weight_shape = infer_shape(weight)
        dense = relay.qnn.op.dense(
            inputs[0],
            weight,
            input_zero_point,
            weight_zero_point,
            input_scale,
            weight_scale,
            units=weight_shape[0],
        )
        bias_var = inputs[1][3]

        return _do_bias_and_requantize(
            dense, bias_var, input_scale, weight_scale, output_scale, output_zero_point, with_relu
        )

    return _impl


def quantized_linear_impl(inputs, input_types):
    func = _linear(False)

    return func(
        [
            inputs[0],
            (
                inputs[3],
                relay.const(inputs[4]),
                relay.const(inputs[5]),
                inputs[6]
            ),
            _get_numpy(inputs[7]).item(),
            _get_numpy(inputs[8]).item(),
            _get_numpy(inputs[1]).item(),
            _get_numpy(inputs[2]).item()
        ],
        None
    )
