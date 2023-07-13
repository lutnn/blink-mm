def calc_output_shape(shape, kernel_size, stride, padding):
    h, w = shape
    out_h = (h + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    out_w = (w + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    return out_h, out_w
