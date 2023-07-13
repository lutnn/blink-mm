from tvm.relay.frontend.qnn_torch import _get_numpy, _op


def quantized_max_pool2d_impl(inputs, input_types):
    data = inputs[0]

    pool_size = _get_numpy(inputs[2]).tolist()
    strides = _get_numpy(inputs[4]).tolist()
    padding = _get_numpy(inputs[3]).tolist()
    dilation = 1

    return _op.nn.max_pool2d(
        data,
        pool_size=pool_size,
        strides=strides,
        dilation=dilation,
        padding=padding,
        layout="NCHW",
    )
