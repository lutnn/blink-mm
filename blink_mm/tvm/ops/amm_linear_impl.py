import tvm
import tvm.relay
from tvm.relay.frontend.common import infer_shape


def amm_linear_impl(inputs, input_types):
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

    linear_op = tvm.relay.nn.amm_linear

    return linear_op(
        data=input,
        bias=bias,
        centroids=centroids,
        lut=lut,
        scale=scale,
        subvec_len=(subvec_len,),
        output_shape=output_shape,
        out_dtype="float32"
    )
