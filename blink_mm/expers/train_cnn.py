import argparse

import torch
from torch.utils.data import Subset

from qat.networks.cnn_wrapper import CNNWrapper
from runner.train import dist_train
from runner.hooks import TensorboardLoggerHook, CheckpointHook

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

from blink_mm.hooks.monitor_temperature_hook import MonitorTemperatureHook
from blink_mm.hooks.switch_quantization_mode_hook import SwitchQuantizationModeHook
from blink_mm.hooks.annealing_temperature_hook import AnnealingTemperatureHook
from blink_mm.expers.utils import get_params, assign_temperature


def extract_base_model_type(model_type):
    if '_' not in model_type:
        return model_type
    prefix = model_type[:model_type.find("_")]
    if prefix not in ["amm", "quantized"]:
        return model_type
    return model_type[model_type.find("_") + 1:]


def strip_state_dict_prefix(ckpt, enforce_prefix=None):
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {
            k.partition('module.')[2]: v
            for k, v in state_dict.items()
        }
    if enforce_prefix is not None and not list(state_dict.keys())[0].startswith(enforce_prefix):
        state_dict = {
            enforce_prefix + k: v
            for k, v in state_dict.items()
        }
    return state_dict


def dist_train_build(rank, world_size, device_id, num_epochs, vars):
    from blink_mm.transforms.transfer import transfer, AMM_PASS
    from blink_mm.data.dispatcher import get_data_config, get_dist_data_loader

    dataset_type = vars["dataset_type"]

    train_data_loader, test_data_loader = get_dist_data_loader(
        rank, world_size, vars["imgs_per_gpu"], dataset_type, vars["root"])
    config = get_data_config(dataset_type)

    if config["num_classes"] == 1:
        Wrapper = RegressionCNNWrapper
    else:
        Wrapper = CNNWrapper

    model_type: str = vars["model_type"]
    base_model_type = extract_base_model_type(model_type)

    device = f"cuda:{device_id}"

    if base_model_type == model_type:
        model = globals()[model_type](**config)
        model = Wrapper(model, device)
        if vars["optimizer"]["name"] == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=vars["lr"],
                momentum=vars["optimizer"]["momentum"],
                weight_decay=vars["optimizer"]["weight_decay"]
            )
        elif vars["optimizer"]["name"] == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=vars["lr"],
                betas=vars["optimizer"]["betas"],
                weight_decay=vars["optimizer"]["weight_decay"]
            )
    else:
        ckpt_path = vars["ckpt_path"]
        if ckpt_path.endswith(".pth"):
            float_model = globals()[base_model_type](**config)
            float_model = Wrapper(float_model, device)
            ckpt = torch.load(vars["ckpt_path"], map_location="cpu")
            ckpt = strip_state_dict_prefix(ckpt, "cnn.")
            float_model.load_state_dict(ckpt)
        else:
            float_model = globals()[base_model_type](
                **config, weights=ckpt_path)
            float_model = Wrapper(float_model, device)
        kwargs = {
            **config,
            "k": vars["num_centroids"],
            "subvec_len": vars["subvec_len"],
            "temperature_config": vars["temperature_config"],
            "fix_weight": vars["fix_weight"],
            "replace_all": vars["replace_all"],
        }
        model = globals()[model_type](**kwargs)
        pass_type = AMM_PASS
        model = Wrapper(model, device)
        transfer(float_model, model,
                 Subset(train_data_loader.dataset, range(1024)), pass_type)

        base_params, centroids_params, temp_params = get_params(model)
        if vars["optimizer"]["name"] == "SGD":
            optimizer = torch.optim.SGD([
                {"params": base_params},
                {"params": centroids_params, "weight_decay": 0},
                {"params": temp_params,
                    "lr": vars["temp_lr"], "weight_decay": 0},
            ],
                lr=vars["lr"],
                momentum=vars["optimizer"]["momentum"],
                weight_decay=vars["optimizer"]["weight_decay"]
            )
        elif vars["optimizer"]["name"] == "Adam":
            optimizer = torch.optim.Adam([
                {"params": base_params},
                {"params": centroids_params, "weight_decay": 0},
                {"params": temp_params,
                    "lr": vars["temp_lr"], "weight_decay": 0},
            ],
                lr=vars["lr"],
                betas=vars["optimizer"]["betas"],
                weight_decay=vars["optimizer"]["weight_decay"]
            )

        if vars.get("temperature", None) is not None:
            assign_temperature(
                vars["temperature_config"], vars["temperature"], temp_params)

    lr_scheduler = vars["lr_scheduler"]
    if lr_scheduler["name"] == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs * len(train_data_loader)
        )
    elif lr_scheduler["name"] == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_scheduler["step_size_epochs"] *
            len(train_data_loader),
            gamma=lr_scheduler.get("gamma", 0.1)
        )
    return model, optimizer, lr_scheduler, train_data_loader, test_data_loader


def get_device_id(rank, world_size, vars):
    device_ids = vars["device_ids"]
    return device_ids[rank % len(device_ids)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train CNN")
    parser.add_argument("--imgs-per-gpu", default=256, type=int)
    parser.add_argument("--root")
    parser.add_argument("--dataset-type", default="cifar10")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--temp-lr", default=1e-1, type=float)
    parser.add_argument("--num-epochs", default=200, type=int)
    parser.add_argument("--work-dir")
    parser.add_argument("--device-ids", default="0")
    parser.add_argument("--num-procs", default=1, type=int)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--model-type", default="amm_resnet18_cifar")
    parser.add_argument(
        "--lr-scheduler", default="{'name':'CosineAnnealingLR'}")
    parser.add_argument(
        "--optimizer", default="{'name':'Adam','betas':(0.9, 0.999),'weight_decay':0}")
    parser.add_argument("--temperature-config", default="inverse")
    parser.add_argument("--temperature", default=[0.065],
                        nargs='*', type=float)
    parser.add_argument("--num-centroids", default=16, type=int)
    parser.add_argument("--subvec-len", default="{'3x3':9,'1x1':4}", type=str)
    parser.add_argument("--fix-weight", action="store_true")
    parser.add_argument("--replace-all", action="store_true")
    parser.add_argument("--log-interval", default=1, type=int)
    parser.add_argument(
        "--checkpoint-hook", default="{'save_best':'accuracy','compare_op':'greater'}")

    args = parser.parse_args()

    checkpoint_hook = eval(args.checkpoint_hook)

    dist_train(
        args.num_procs, args.work_dir, args.num_epochs,
        get_device_id, dist_train_build, vars={
            "imgs_per_gpu": args.imgs_per_gpu,
            "root": args.root,
            "lr": args.lr,
            "temp_lr": args.temp_lr,
            "ckpt_path": args.ckpt_path,
            "model_type": args.model_type,
            "lr_scheduler": eval(args.lr_scheduler),
            "optimizer": eval(args.optimizer),
            "dataset_type": args.dataset_type,
            "device_ids": list(map(int, args.device_ids.split(','))),

            # ablation study
            "temperature_config": args.temperature_config,
            "temperature": args.temperature[0] if (args.temperature is not None and len(args.temperature) == 1) else None,
            "num_centroids": args.num_centroids,
            "subvec_len": eval(args.subvec_len),
            "fix_weight": args.fix_weight,
            "replace_all": args.replace_all,
        },
        hooks=[
            CheckpointHook(
                save_best=checkpoint_hook["save_best"],
                compare_op=checkpoint_hook["compare_op"]
            ),
            SwitchQuantizationModeHook(5000),
            TensorboardLoggerHook(args.log_interval),
            MonitorTemperatureHook(),
            *([AnnealingTemperatureHook(*args.temperature)]
              if args.temperature_config == "manual" else [])
        ]
    )
