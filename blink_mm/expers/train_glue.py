import os.path as osp
import argparse

import torch
from torch.utils.data import Subset

from transformers import AdamW

from runner.train import dist_train
from runner.hooks import TensorboardLoggerHook, CheckpointHook

from blink_mm.networks.seq_models.amm_bert import AMMBERT
from blink_mm.networks.seq_models.bert import BERT
from blink_mm.networks.seq_models.seq_classification_network import SeqClassificationNetwork
from blink_mm.hooks.monitor_temperature_hook import MonitorTemperatureHook
from blink_mm.data.glue import CONFIG


def dist_train_build(rank, world_size, device_id, num_epochs, vars):
    from blink_mm.data.glue import get_dist_train_data_loader, get_dist_test_data_loaders, get_tokenizer

    model_type = vars["model_type"]
    dataset_type = vars["dataset_type"]
    num_labels = CONFIG["num_labels"][dataset_type]
    tokenizer = get_tokenizer()
    train_data_loader = get_dist_train_data_loader(
        rank, world_size, tokenizer, dataset_type, vars["batch_size_per_gpu"]
    )
    test_data_loaders = get_dist_test_data_loaders(
        rank, world_size, tokenizer, dataset_type, vars["batch_size_per_gpu"]
    )

    device = f"cuda:{device_id}"

    if model_type == "bert":
        num_hidden_layers = vars.get("num_hidden_layers", 4)
        model = BERT(num_labels=num_labels,
                     num_hidden_layers=num_hidden_layers)
        model = SeqClassificationNetwork(model, device)
    elif model_type == "amm_bert":
        num_layers_to_replace = vars.get("num_layers_to_replace", 3)
        num_hidden_layers = vars.get("num_hidden_layers", 4)
        k = vars.get("k", 16)
        subvec_len = vars.get("subvec_len", 16)
        model = AMMBERT(
            num_labels=num_labels,
            num_hidden_layers=num_hidden_layers,
            num_layers_to_replace=num_layers_to_replace,
            k=k, subvec_len=subvec_len
        ).to(device)
        ckpt = torch.load(vars["ckpt_path"], map_location=device)
        ckpt = {
            k.partition('seq_model.')[2]: v
            for k, v in ckpt["state_dict"].items()
            if "transformer.classifier" not in k
        }
        model.load_state_dict(ckpt, strict=False)
        model = SeqClassificationNetwork(model, device)

    no_decay = ['bias', 'LayerNorm.weight']
    temperature = "inverse_temperature_logit"
    parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and
                not temperature in n
            ], 'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ], 'weight_decay': 0.0
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if temperature in n
            ], 'weight_decay': 0.0, "lr": vars["temp_lr"]
        }
    ]

    optimizer = AdamW(parameters, lr=vars["lr"])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, num_epochs * len(train_data_loader), gamma=1
    )
    return model, optimizer, lr_scheduler, train_data_loader, test_data_loaders


def get_device_id(rank, world_size, vars):
    device_ids = vars["device_ids"]
    return device_ids[rank % len(device_ids)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train GLUE")
    parser.add_argument("--batch-size-per-gpu", default=32, type=int)
    parser.add_argument("--temp-lr", default=1e-1, type=float)
    parser.add_argument("--work-dir")
    parser.add_argument("--device-ids", default="0")
    parser.add_argument("--num-procs", default=1, type=int)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--model-type", default="amm_bert")

    args = parser.parse_args()

    assert args.model_type in ["amm_bert", "bert"]

    for dataset_type in list(CONFIG["num_labels"].keys()):
        for lr in [5e-5, 4e-5, 3e-5, 2e-5]:
            vars = {
                "batch_size_per_gpu": args.batch_size_per_gpu,
                "lr": lr,
                "temp_lr": args.temp_lr,
                "dataset_type": dataset_type,
                "device_ids": list(map(int, args.device_ids.split(','))),
                "model_type": args.model_type,
                "ckpt_path": args.ckpt_path,
            }
            dist_train(
                args.num_procs,
                osp.join(args.work_dir, dataset_type, f"lr={lr}"),
                3,
                get_device_id, dist_train_build, vars=vars,
                hooks=[
                    CheckpointHook(save_best=CONFIG["metrics"][dataset_type]),
                    TensorboardLoggerHook(1),
                    MonitorTemperatureHook()
                ]
            )
