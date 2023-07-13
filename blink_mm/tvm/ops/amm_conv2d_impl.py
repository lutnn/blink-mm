import tvm
import tvm.relay
from tvm.relay.frontend.common import infer_shape


def amm_conv2d_impl(inputs, input_types):
    centroids_shape = infer_shape(inputs[2])

    input = inputs[0]
    bias = inputs[1]
    # from (nc, num_centroids, subvec_len)
    # to (nc, num_centroids // 8, subvec_len, 8)
    nc, num_centroids, subvec_len = centroids_shape
    centroids = tvm.relay.transpose(
        tvm.relay.reshape(inputs[2], (nc, num_centroids // 8, 8, subvec_len)),
        (0, 1, 3, 2)
    )
    lut = tvm.relay.transpose(inputs[3], (2, 0, 1))
    scale = tvm.relay.const(tvm.nd.array([inputs[4]]), dtype="float32")
    output_shape = inputs[5].data.numpy().tolist()
    kernel_size = inputs[6].data.numpy().tolist()
    strides = inputs[7].data.numpy().tolist()
    padding = inputs[8].data.numpy().tolist()

    conv_op = tvm.relay.nn.amm_conv2d
    return conv_op(
        data=input,
        bias=bias,
        centroids=centroids,
        lut=lut,
        scale=scale,
        subvec_len=(subvec_len,),
        output_shape=output_shape,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        out_layout="",
        out_dtype="float32"
    )
