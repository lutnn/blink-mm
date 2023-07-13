from glob import glob

import pandas as pd
from tqdm import tqdm

import torch

from qat.networks.cnn_wrapper import CNNWrapper
from qat.export.utils import fetch_module_by_name, replace_module_by_name

from blink_mm.expers.train_cifar import strip_state_dict_prefix
from blink_mm.expers.utils import evaluate_model
from blink_mm.networks.pytorch_resnet_cifar10.resnet import *
from blink_mm.networks.third_party_resnet.amm_resnet_cifar import *
from blink_mm.networks.third_party_resnet.madness_resnet_cifar import *
from blink_mm.ops.maddness.maddness_conv2d import MaddnessConv2d
from blink_mm.ops.maddness.maddness_linear import MaddnessLinear
from blink_mm.ops.amm_conv2d import AMMConv2d
from blink_mm.ops.amm_linear import AMMLinear


def evaluate_mse(float_model, model_to_replace, test_data_loader, name):
    import torch.nn.functional as F

    tot_mse = 0
    num_iters = 0

    for data_batch in tqdm(test_data_loader, f"Evaluating MSE of {name}"):
        x = float_model(*float_model.get_input_tuple(float_model, data_batch))
        y = model_to_replace(
            *model_to_replace.get_input_tuple(model_to_replace, data_batch))
        mse = F.mse_loss(x, y).item()
        tot_mse += mse
        num_iters += 1

    return tot_mse / num_iters


def eval_per_layer_cifar10(model_type, madness_ckpt_path, root):
    from blink_mm.data.cifar import get_test_data_loader

    test_data_loader = get_test_data_loader(128, "cifar10", root)

    base_model_type = model_type[model_type.find("_") + 1:]

    device = "cuda:0"

    ckpt_path = f"blink_mm/networks/pytorch_resnet_cifar10/pretrained_models/{base_model_type}*"
    ckpt_path = glob(ckpt_path)[0]
    model_to_replace = globals()[base_model_type]()
    model_to_replace.load_state_dict(
        strip_state_dict_prefix(torch.load(ckpt_path)))
    model_to_replace = CNNWrapper(model_to_replace, device)
    float_model = globals()[base_model_type]()
    float_model = CNNWrapper(float_model, device)
    float_model.load_state_dict(model_to_replace.state_dict())

    model = globals()[model_type](replace_linear=True)
    model = CNNWrapper(model, device)
    model.load_state_dict(torch.load(madness_ckpt_path, map_location=device))

    name_list = []

    for name, module in model.named_modules():
        if isinstance(module, MaddnessConv2d) or \
                isinstance(module, MaddnessLinear) or \
                isinstance(module, AMMConv2d) or \
                isinstance(module, AMMLinear):
            name_list.append(name)

    name_list = reversed(name_list)

    df = pd.DataFrame(columns=["layer", "accuracy", "mse"])

    for name in name_list:
        replace_module_by_name(
            model_to_replace, name, fetch_module_by_name(model, name))
        evaluation = evaluate_model(model_to_replace, test_data_loader, name)
        accuracy = evaluation["accuracy"]
        mse = evaluate_mse(float_model, model_to_replace,
                           test_data_loader, name)
        df = pd.concat([
            df, pd.DataFrame([[name, accuracy, mse]], columns=["layer", "accuracy", "mse"])],
            ignore_index=True)

    df.to_csv("per_layer_accuracy.csv", index=False)


if __name__ == "__main__":
    eval_per_layer_cifar10(
        "amm_resnet20",
        "/mnt/elasticedge1/xiaohu/ckpts/madness/pq-resnet20-replace-linear.pth",
        "./datasets"
    )
