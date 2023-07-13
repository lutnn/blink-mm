import os
import os.path as osp
import argparse

import pandas as pd
from tqdm import tqdm
import torch
from torchvision.models.resnet import *

from opcounter.counter import counter
from opcounter.hooks import *

from qat.ops import QuantizedOperator

# cifar models
from blink_mm.networks.resnet_large_cifar.resnet import *  # resnet18_cifar
from blink_mm.networks.resnet_large_cifar.amm_resnet import *  # amm_resnet18_cifar
from blink_mm.networks.vgg_cifar.vgg import *  # vgg11_cifar
from blink_mm.networks.vgg_cifar.amm_vgg import *  # amm_vgg11_cifar
from blink_mm.networks.senet_cifar.senet import *  # senet18_cifar
from blink_mm.networks.senet_cifar.amm_senet import *  # amm_senet18_cifar

# imagenet models
from torchvision.models.resnet import *  # resnet18
from blink_mm.networks.resnet_large_imagenet.amm_resnet import *  # amm_resnet18
from blink_mm.networks.vgg_imagenet.vgg import *  # vgg11_bn
from blink_mm.networks.vgg_imagenet.amm_vgg import *  # amm_vgg11_bn
from blink_mm.networks.senet_imagenet.senet import *  # senet18
from blink_mm.networks.senet_imagenet.amm_senet import *  # amm_senet18

from blink_mm.networks.seq_models.seq_classification_network import SeqClassificationNetwork
from blink_mm.networks.seq_models.bert import BERT
from blink_mm.networks.seq_models.amm_bert import AMMBERT
from blink_mm.count.hooks import *
from blink_mm.count.fvcore_handlers import *


HOOKS = [
    Conv2DForwardHook(),
    LinearForwardHook(),
    AMMConv2dForwardHook(),
    MaddnessConv2DForwardHook(),
    QuantizedConv2dBatchNorm2dReLUForwardHook(),
    QuantizedAMMConv2dBatchNorm2dReLUForwardHook(),
    QuantizedLinearForwardHook(),
]


def switch_quantization_mode(model):
    for module in model.modules():
        if isinstance(module, QuantizedOperator):
            module.num_batches_tracked.data.copy_(torch.tensor(1))
            module.running_min.data.copy_(torch.tensor(0))
            module.running_max.data.copy_(torch.tensor(1))
            module.activation_quantization = True


def count_cifar10():
    df = pd.DataFrame()
    for model_type, kwargs in tqdm([
        ["resnet18_cifar", {}],
        ["amm_resnet18_cifar", {"k": 8}],
        ["amm_resnet18_cifar", {"k": 16}],
        ["senet18_cifar", {}],
        ["amm_senet18_cifar", {"k": 8}],
        ["amm_senet18_cifar", {"k": 16}],
        ["vgg11_cifar", {}],
        ["amm_vgg11_cifar", {"k": 8}],
        ["amm_vgg11_cifar", {"k": 16}],
    ]):
        model = globals()[model_type](**kwargs)
        dst = counter(model, (torch.randn(1, 3, 32, 32),), HOOKS)
        df = pd.concat([df, pd.DataFrame.from_records([{
            "model": model_type + str(kwargs),
            **dst,
        }])], ignore_index=True)

    df = df.fillna(0)
    return df


def count_imagenet():
    df = pd.DataFrame()

    for model_type, kwargs in tqdm([
        ["resnet18", {}],
        ["amm_resnet18", {"k": 8}],
        ["amm_resnet18", {"k": 16}],
        ["senet18", {}],
        ["amm_senet18", {"k": 8}],
        ["amm_senet18", {"k": 16}],
        ["vgg11_bn", {}],
        ["amm_vgg11_bn", {"k": 8}],
        ["amm_vgg11_bn", {"k": 16}],
    ]):
        model = globals()[model_type](**kwargs)
        dst = counter(model, (torch.randn(1, 3, 224, 224),), HOOKS)
        df = pd.concat([df, pd.DataFrame.from_records([{
            "model": model_type + str(kwargs),
            **dst,
        }])], ignore_index=True)

    df = df.fillna(0)
    return df


def bert_half(num_labels):
    return SeqClassificationNetwork(
        BERT(num_labels, num_hidden_layers=6, torchscript=True),
        device="cpu"
    )


def amm_bert_half(num_labels, subvec_len):
    return SeqClassificationNetwork(
        AMMBERT(
            num_labels,
            subvec_len=subvec_len,
            num_hidden_layers=6,
            torchscript=True
        ),
        device="cpu"
    )


def count_bert():
    from fvcore.nn import FlopCountAnalysis
    from qat.export.export import _get_model_to_export
    from blink_mm.data.glue import CONFIG, get_tokenizer, get_dist_data_loader
    from blink_mm.ops.amm_linear import AMMLinear
    from blink_mm.transforms.export.handlers import AMMLinearHandler

    name = "sst2"
    tokenizer = get_tokenizer()
    data_loader = get_dist_data_loader(
        0, 1, tokenizer, name, "train", 1, False)

    df = pd.DataFrame()

    for model_type, kwargs in [
        ["bert_half", {}],
        ["amm_bert_half", {'subvec_len': 32}],  # subvec len = 32
        ["amm_bert_half", {'subvec_len': 16}],  # subvec len = 16
    ]:
        model = globals()[model_type](CONFIG["num_labels"][name], **kwargs)

        for data_batch in data_loader:
            input_tuple = model.get_input_tuple(model, data_batch)
            model = _get_model_to_export(model, input_tuple, {
                AMMLinear: AMMLinearHandler
            })
            flops = FlopCountAnalysis(model, input_tuple).set_op_handle(
                "prim::PythonOp.AMMLinearFn", amm_linear_fn_flop_jit
            )
            df = pd.concat([df, pd.DataFrame.from_records([{
                "model": model_type + str(kwargs),
                "ops": flops.total(),
            }])], ignore_index=True)
            break

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    output_dir = osp.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.concat([count_cifar10(), count_imagenet()], ignore_index=True)
    df.to_csv(osp.join(output_dir, "count_cnn.csv"), index=False)

    df = count_bert()
    df.to_csv(osp.join(output_dir, "count_bert.csv"), index=False)
