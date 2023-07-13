from tvm.relay.frontend.qnn_torch import infer_shape

from blink_mm.tvm.ops.amm_conv2d_impl import amm_conv2d_impl
from blink_mm.tvm.ops.amm_linear_impl import amm_linear_impl
from blink_mm.tvm.ops.quantize_impl import quantize_impl
from blink_mm.tvm.ops.quantized_conv2d_impl import quantized_conv2d_impl
from blink_mm.tvm.ops.quantized_max_pool2d_impl import quantized_max_pool2d_impl
from blink_mm.tvm.ops.quantized_add_impl import quantized_add_impl
from blink_mm.tvm.ops.quantized_relu_impl import quantized_relu_impl
from blink_mm.tvm.ops.quantized_global_avg_pool_impl import quantized_global_avg_pool_impl
from blink_mm.tvm.ops.quantized_linear_impl import quantized_linear_impl
from blink_mm.tvm.ops.quantized_amm_conv2d_impl import quantized_amm_conv2d_impl


# a dirty hack
def amm_op_impl(inputs, input_types):
    # amm
    if len(inputs) == 9:
        return amm_conv2d_impl(inputs, input_types)
    if len(inputs) == 6 and len(infer_shape(inputs[2])) == 3:
        return amm_linear_impl(inputs, input_types)

    # quantized operators
    if len(inputs) == 3:
        return quantize_impl(inputs, input_types)
    if len(inputs) == 15 and input_types[6] == "float32":
        return quantized_conv2d_impl(inputs, input_types)
    if len(inputs) == 5 and infer_shape(inputs[1])[0] > 1:
        return quantized_max_pool2d_impl(inputs, input_types)
    if len(inputs) == 8:
        return quantized_add_impl(inputs, input_types)
    if len(inputs) == 5 and infer_shape(inputs[1])[0] == 1:
        return quantized_relu_impl(inputs, input_types)
    if len(inputs) == 6 and len(infer_shape(inputs[1])) == 1:
        return quantized_global_avg_pool_impl(inputs, input_types)
    if len(inputs) == 10:
        return quantized_linear_impl(inputs, input_types)

    # quantized amm
    if len(inputs) == 15 and input_types[6] == "int8":
        return quantized_amm_conv2d_impl(inputs, input_types)
