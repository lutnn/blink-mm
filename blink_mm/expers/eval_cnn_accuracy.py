import argparse
import os.path as osp
from glob import glob
import csv

import torch
from tqdm import tqdm

from qat.networks.cnn_wrapper import CNNWrapper

from blink_mm.networks.regression_cnn_wrapper import RegressionCNNWrapper

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


def get_configs():
    configs = {}
    for dataset_type in ["cifar10", "gtsrb", "speech_commands", "svhn"]:
        configs[dataset_type] = {
            "metric": "accuracy",
            "models": ["resnet18_cifar", "vgg11_cifar", "senet18_cifar"],
            "root": "./datasets",
        }
    configs["imagenet"] = {
        "metric": "accuracy",
        "models": [{
            "name": "resnet18",
            "weights": "IMAGENET1K_V1",
            "number": 0.69758
        }, "vgg11_bn", "senet18"],
        "root": "./datasets/imagenet-raw-data"
    }
    return configs


def load_ckpt(model, dataset, metric, ckpt_folder):
    if model.startswith("maddness"):
        model_folder = osp.join(ckpt_folder, "maddness")
        model_path = osp.join(model_folder, model + "-" + dataset + ".pth")
    else:
        model_folder = osp.join(ckpt_folder, model + "-" + dataset)
        model_paths = glob(
            osp.join(model_folder, f"default_best_{metric}_epoch_*.pth"))
        assert len(model_paths) == 1, model_folder
        model_path = model_paths[0]
    return torch.load(model_path, map_location="cpu")


def fast_evaluate(model, dataset, metric, ckpt_folder):
    ckpt = load_ckpt(model, dataset, metric, ckpt_folder)
    return ckpt["meta"]["default"][metric]


def evaluate(model, dataset, metric, ckpt_folder, root):
    from blink_mm.data.dispatcher import get_data_config, get_dist_data_loader

    data_config = get_data_config(dataset)
    if data_config["num_classes"] == 1:
        Wrapper = RegressionCNNWrapper
    else:
        Wrapper = CNNWrapper

    ckpt = load_ckpt(model, dataset, metric, ckpt_folder)
    model = globals()[model](**data_config)
    model = Wrapper(model, "cpu")
    model.load_state_dict(ckpt["state_dict"])

    _, test_data_loader = get_dist_data_loader(0, 1, 16, dataset, root)
    model.eval()
    xs, ys = [], []
    for data_batch in tqdm(test_data_loader):
        with torch.no_grad():
            x, y = model.val_step(model, data_batch, None)
            xs.append(x)
            ys.append(y)
    return model.evaluate(xs, ys)[metric]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--ckpt-folder")
    parser.add_argument("--output-csv")
    parser.add_argument("--fast-mode", action="store_true")
    args = parser.parse_args()

    ckpt_folder = osp.expanduser(args.ckpt_folder)

    configs = get_configs()

    f = open(args.output_csv, "w")
    writer = csv.writer(f)
    writer.writerow(["model", "number", "dataset", "metric"])

    for dataset in [
        "cifar10", "gtsrb", "speech_commands", "svhn", "imagenet"
    ]:
        for model in configs[dataset]["models"]:
            metric = configs[dataset]["metric"]
            torchvision_number = None
            if isinstance(model, dict):
                torchvision_number = model["number"]
                model = model["name"]

            for prefix in ["", "amm_", "maddness_"]:
                if torchvision_number is not None and prefix == "":
                    number = torchvision_number
                elif args.fast_mode:
                    number = fast_evaluate(
                        prefix + model, dataset, metric, ckpt_folder)
                else:
                    number = evaluate(
                        prefix + model, dataset, metric,
                        ckpt_folder, configs[dataset]["root"]
                    )
                writer.writerow([prefix + model, number, dataset, metric])
                f.flush()

    f.close()
