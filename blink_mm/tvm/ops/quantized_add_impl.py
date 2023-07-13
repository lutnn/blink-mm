from tvm import relay
from tvm.relay.frontend.qnn_torch import _binop


def quantized_add_impl(inputs, input_types):
    func = _binop(relay.qnn.op.add)

    return func(
        [
            inputs[0],
            inputs[3],
            inputs[1],
            inputs[2],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7]
        ],
        None
    )
