import torch


class AMMConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor, bias: torch.Tensor,
        centroids: torch.Tensor, lut: torch.Tensor, s: torch.Tensor,
        output_shape: torch.Tensor, kernel_size: torch.Tensor, stride: torch.Tensor, padding: torch.Tensor
    ) -> torch.Tensor:
        return torch.empty(output_shape.tolist()).to(torch.float32)

    @staticmethod
    def symbolic(
        g,
        input: torch.Tensor, bias: torch.Tensor,
        centroids: torch.Tensor, lut: torch.Tensor, s: torch.Tensor,
        output_shape: torch.Tensor, kernel_size: torch.Tensor, stride: torch.Tensor, padding: torch.Tensor
    ):
        return g.op(
            "com.microsoft::DPQConv2d",
            input, bias, centroids, lut, s,
            kernel_size_i=kernel_size.tolist(), stride_i=stride.tolist(), padding_i=padding.tolist()
        )


class AMMLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input, bias,
        centroids, lut, s,
        output_shape
    ):
        return torch.empty(output_shape.tolist()).to(torch.float32)

    @staticmethod
    def symbolic(
        g,
        input: torch.Tensor, bias: torch.Tensor,
        centroids: torch.Tensor, lut: torch.Tensor, s: torch.Tensor,
        output_shape: torch.Tensor
    ):
        return g.op(
            "com.microsoft::DPQLinear",
            input, bias, centroids, lut, s
        )


class QuantizedAMMConv2dBatchNorm2dReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        int_x,
        input_scale,
        input_zero_point,
        int_centroids,
        centroids_scale,
        centroids_zero_point,
        int_lut,
        lut_scale,
        int_bias,
        output_scale,
        output_zero_point,
        output_dtype,
        out_shape,
        kernel_size,
        padding,
        stride,
    ):
        return torch.empty(out_shape.tolist()).to(output_dtype)

    @staticmethod
    def symbolic(
        g,
        int_x,
        input_scale,
        input_zero_point,
        int_centroids,
        centroids_scale,
        centroids_zero_point,
        int_lut,
        lut_scale,
        int_bias,
        output_scale,
        output_zero_point,
        output_dtype,
        out_shape,
        kernel_size,
        padding,
        stride,
    ):
        return g.op(
            "com.microsoft::DPQQuantConv2d",
            int_x,
            input_scale,
            input_zero_point,
            int_centroids,
            centroids_scale,
            centroids_zero_point,
            int_lut,
            lut_scale,
            int_bias,
            output_scale,
            output_zero_point,
            kernel_size_i=kernel_size.tolist(),
            stride_i=stride.tolist(),
            padding_i=padding.tolist()
        )
