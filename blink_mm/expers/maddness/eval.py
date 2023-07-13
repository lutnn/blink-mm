import argparse
import os.path as osp
from glob import glob

import torch
from torch.utils.data import Subset

from tqdm import tqdm

from qat.networks.cnn_wrapper import CNNWrapper

from blink_mm.networks.regression_cnn_wrapper import RegressionCNNWrapper

# cifar models
from blink_mm.networks.resnet_large_cifar.resnet import *  # resnet18_cifar
# maddness_resnet18_cifar
from blink_mm.networks.resnet_large_cifar.maddness_resnet import *
from blink_mm.networks.vgg_cifar.vgg import *  # vgg11_cifar
from blink_mm.networks.vgg_cifar.maddness_vgg import *  # maddness_vgg11_cifar
from blink_mm.networks.senet_cifar.senet import *  # senet18_cifar
from blink_mm.networks.senet_cifar.maddness_senet import *  # maddness_senet18_cifar

# imagenet models
from torchvision.models.resnet import *  # resnet18
from blink_mm.networks.resnet_large_imagenet.maddness_resnet import *  # maddness_resnet18
from blink_mm.networks.vgg_imagenet.vgg import *  # vgg11_bn
from blink_mm.networks.vgg_imagenet.maddness_vgg import *  # maddness_vgg11_bn
from blink_mm.networks.senet_imagenet.senet import *  # senet18
from blink_mm.networks.senet_imagenet.maddness_senet import *  # maddness_senet18

from blink_mm.transforms.transfer import transfer, MADDNESS_PASS
from blink_mm.expers.train_cnn import strip_state_dict_prefix


def evaluate_model(model, test_data_loader):
    model.eval()
    xs, ys = [], []
    for data_batch in tqdm(test_data_loader):
        with torch.no_grad():
            x, y = model.val_step(model, data_batch, None)
            xs.append(x)
            ys.append(y)
    return model.evaluate(xs, ys)


def eval_maddness(model_type, dataset_type, ckpt_path, save_ckpt_path, metric, root):
    from blink_mm.data.dispatcher import get_data_config, get_dist_data_loader

    base_model_type = model_type[model_type.find("_") + 1:]

    config = get_data_config(dataset_type)
    if config["num_classes"] == 1:
        Wrapper = RegressionCNNWrapper
    else:
        Wrapper = CNNWrapper
    device = "cuda:0"
    if not ckpt_path.endswith(".pth"):
        float_model = globals()[base_model_type](**config, weights=ckpt_path)
        float_model = Wrapper(float_model, device)
    else:
        float_model = globals()[base_model_type](**config)
        float_model = Wrapper(float_model, device)
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt = strip_state_dict_prefix(ckpt, "cnn.")
        float_model.load_state_dict(ckpt)

    model = globals()[model_type](**config)
    model = Wrapper(model, device)

    train_data_loader, test_data_loader = get_dist_data_loader(
        0, 1, 32, dataset_type, root)
    dataset = train_data_loader.dataset
    dataset = Subset(dataset, range(1024))

    transfer(float_model, model, dataset, MADDNESS_PASS)

    results = evaluate_model(model, test_data_loader)
    if save_ckpt_path is not None:
        torch.save({
            "meta": {metric: results[metric]},
            "state_dict": model.state_dict(),
        }, save_ckpt_path)

    return results


def get_configs():
    configs = {}
    for dataset_type in ["cifar10", "cifar100", "svhn", "gtsrb", "speech_commands"]:
        configs[dataset_type] = {
            "metric": "accuracy",
            "models": ["maddness_resnet18_cifar", "maddness_vgg11_cifar", "maddness_senet18_cifar"],
            "root": "./datasets",
        }
    configs["utk_face"] = {
        "metric": "mae",
        "models": ["maddness_resnet18"],
        "root": "datasets",
    }
    configs["imagenet"] = {
        "metric": "accuracy",
        "models": [{
            "name": "maddness_resnet18",
            "weights": "IMAGENET1K_V1"
        },
            "maddness_vgg11_bn", "maddness_senet18"
        ],
        "root": "datasets/imagenet-raw-data"
    }
    return configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Eval MADDNESS")
    parser.add_argument("--root-ckpt-folder")
    parser.add_argument("--save-ckpt-folder")
    args = parser.parse_args()

    configs = get_configs()

    for dataset_type in configs.keys():
        for model_type in configs[dataset_type]["models"]:
            if isinstance(model_type, dict):
                ckpt_path = model_type["weights"]
                model_type = model_type["name"]
                base_model_type = model_type[model_type.find("_") + 1:]
            else:
                base_model_type = model_type[model_type.find("_") + 1:]
                ckpt_folder = osp.join(
                    args.root_ckpt_folder, base_model_type + "-" + dataset_type)
                ckpt_path = glob(osp.join(
                    ckpt_folder, f"default_best_{configs[dataset_type]['metric']}_*.pth"))
                assert len(ckpt_path) == 1
                ckpt_path = ckpt_path[0]

            results = eval_maddness(
                model_type,
                dataset_type,
                ckpt_path,
                osp.join(args.save_ckpt_folder, "maddness_" +
                         base_model_type + "-" + dataset_type + ".pth"),
                configs[dataset_type]['metric'],
                configs[dataset_type]["root"]
            )

            print(dataset_type, model_type)
            print(results)
