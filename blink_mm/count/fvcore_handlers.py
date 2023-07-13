from typing import List, Any

import numpy as np

from fvcore.nn.jit_handles import get_shape


def amm_linear_fn_flop_jit(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for AMMLinearFn.
    """
    input_shapes = [get_shape(v) for v in inputs]
    out_features = input_shapes[1][0]
    ncodebooks, k, subvec_len = input_shapes[2]
    n = np.prod(input_shapes[0][:-1])
    flops = ncodebooks * n * k * subvec_len + ncodebooks * n * out_features
    return flops
