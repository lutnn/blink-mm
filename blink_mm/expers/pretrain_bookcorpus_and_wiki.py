import os.path as osp
import argparse

import torch
from torch.utils.data import Subset

from transformers import AdamW

from runner.train import dist_train
from runner.hooks import TensorboardLoggerHook, CheckpointHook

from blink_mm.networks.seq_models.amm_bert import AMMBERT
from blink_mm.networks.seq_models.bert import BERT
from blink_mm.networks.seq_models.mlm import MLM
from blink_mm.hooks.monitor_temperature_hook import MonitorTemperatureHook


def dist_train_build(rank, world_size, device_id, num_epochs, vars):
    from blink_mm.data.bookcorpus_and_wiki.data import get_dist_train_data_loader, get_tokenizer
    from blink_mm.transforms.transfer import transfer, AMM_PASS

    tokenizer = get_tokenizer()
    train_data_loader = get_dist_train_data_loader(
        rank, world_size, tokenizer, vars["batch_size_per_gpu"], vars["dataset_path"]
    )

    device = f"cuda:{device_id}"

    float_model = MLM(BERT(num_labels=1, num_hidden_layers=4), device)

    num_layers_to_replace = vars.get("num_layers_to_replace", 3)
    k = vars.get("k", 16)
    subvec_len = vars.get("subvec_len", 16)

    model = MLM(AMMBERT(
        num_labels=1,
        num_hidden_layers=4,
        num_layers_to_replace=num_layers_to_replace,
        k=k, subvec_len=subvec_len
    ), device)
    pass_type = AMM_PASS
    transfer(
        float_model, model,
        Subset(train_data_loader.dataset,
               range(min(1024, len(train_data_loader.dataset)))),
        pass_type
    )

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

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_data_loader)
    )
    return model, optimizer, lr_scheduler, train_data_loader, {}


def get_device_id(rank, world_size, vars):
    device_ids = vars["device_ids"]
    return device_ids[rank % len(device_ids)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train GLUE")
    parser.add_argument("--batch-size-per-gpu", default=256, type=int)
    parser.add_argument("--temp-lr", default=1e-1, type=float)
    parser.add_argument("--work-dir")
    parser.add_argument("--dataset-path")
    parser.add_argument("--device-ids", default="0")
    parser.add_argument("--num-procs", default=1, type=int)

    args = parser.parse_args()

    vars = {
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "lr": 1e-4,
        "temp_lr": args.temp_lr,
        "dataset_path": args.dataset_path,
        "device_ids": list(map(int, args.device_ids.split(','))),
        "find_unused_parameters": True
    }

    dist_train(
        args.num_procs,
        args.work_dir,
        10,
        get_device_id, dist_train_build, vars=vars,
        hooks=[
            CheckpointHook(),
            TensorboardLoggerHook(1),
            MonitorTemperatureHook()
        ]
    )
