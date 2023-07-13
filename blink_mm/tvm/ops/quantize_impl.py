import tvm
import tvm.relay
from tvm.relay.frontend.common import infer_shape


def quantize_impl(inputs, _):
    input_shape = infer_shape(inputs[0])
    scale_shape = infer_shape(inputs[1])
    assert len(scale_shape) == 1

    return tvm.relay.qnn.op.quantize(
        inputs[0],
        tvm.relay.repeat(inputs[1], input_shape[0], axis=0),
        tvm.relay.repeat(
            tvm.relay.cast(inputs[2], "int32"), input_shape[0], axis=0),
        out_dtype="int8",
        axis=0
    )
