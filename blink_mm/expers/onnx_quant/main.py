import argparse

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from blink_mm.expers.onnx_quant.data_reader import DataReader


def get_data_reader(input_names, calibrate_dataset, root):
    from blink_mm.data.cifar import get_train_data_loader as cifar10_get_data_loader
    from qat.data.imagenet import get_dist_train_data_loader as imagenet_get_data_loader

    if calibrate_dataset == "imagenet":
        data_loader = imagenet_get_data_loader(0, 1, 1, root)
    elif calibrate_dataset == "cifar10":
        data_loader = cifar10_get_data_loader(1, "cifar10", root)

    return DataReader(input_names, data_loader, 1024)


def get_input_names(input_model_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(input_model_path, None)
    return {
        input.name: i
        for i, input in enumerate(sess.get_inputs())
    }


def get_output_model_path(input_model_path):
    import os.path as osp
    basename = osp.basename(input_model_path)
    dirname = osp.dirname(input_model_path)
    return osp.join(dirname, "quant-" + basename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--input-model-path")
    parser.add_argument("--calibrate-dataset", choices=["imagenet", "cifar10"])
    parser.add_argument("--root")

    args = parser.parse_args()

    input_names = get_input_names(args.input_model_path)
    data_reader = get_data_reader(
        input_names, args.calibrate_dataset, args.root)

    quantize_static(
        args.input_model_path,
        get_output_model_path(args.input_model_path),
        data_reader,
        quant_format=QuantFormat.QOperator,
        per_channel=True,
        weight_type=QuantType.QInt8,
    )
