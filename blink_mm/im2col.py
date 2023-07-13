from typing import Union

import torch
import torch.nn.functional as F

from qat.ops import QuantizedTensor

import numpy as np


def im2col_get_indices(x_shape, kernel_size, stride, padding):
    """
    Returns index matrices in order to transform our input image into a matrix.
    """
    # get input size
    _, c, h, w = x_shape

    # get output size
    out_h = (h + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    out_w = (w + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

    # Compute matrix of index i

    # Level 1 vector.
    level_1 = np.repeat(np.arange(kernel_size[0]), kernel_size[1])
    # Duplicate for the other channels.
    level_1 = np.tile(level_1, c)
    # Create a vector with an increase by 1 at each level.
    every_levels = stride[0] * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level_1.reshape(-1, 1) + every_levels.reshape(1, -1)

    # Compute matrix of index j

    # Slide 1 vector.
    slide_1 = np.tile(np.arange(kernel_size[1]), kernel_size[0])
    # Duplicate for the other channels.
    slide_1 = np.tile(slide_1, c)
    # Create a vector with an increase by 1 at each slide.
    every_slides = stride[1] * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide_1.reshape(-1, 1) + every_slides.reshape(1, -1)

    # Compute matrix of index d

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(c),
                  kernel_size[0] * kernel_size[1]).reshape(-1, 1)

    return i, j, d


def im2col(x, kernel_size, stride, padding, constant_values=0):
    """
    Transforms our input image into a matrix.
    """
    # Padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding[0], padding[0]),
                          (padding[1], padding[1])), mode='constant', constant_values=constant_values)
    i, j, d = im2col_get_indices(
        x.shape, kernel_size, stride, padding)
    # Multi-dimensional arrays indexing.
    cols = x_padded[:, d, i, j]
    return cols


def unfold(x: Union[QuantizedTensor, torch.Tensor], kernel_size, stride, padding):
    if isinstance(x, QuantizedTensor):
        q = im2col(x.q.detach().cpu().numpy(), kernel_size, stride, padding,
                   x.z.detach().cpu().numpy())
        q = torch.tensor(q).to(x.q.device, x.q.dtype)
        if x.r is not None:
            r = im2col(x.r.detach().cpu().numpy(),
                       kernel_size, stride, padding)
            r = torch.tensor(r).to(x.q.device, x.r.dtype)
        else:
            r = None
        return QuantizedTensor(q, x.s, x.z, r)
    else:
        return F.unfold(x, kernel_size, (1, 1), padding, stride)
