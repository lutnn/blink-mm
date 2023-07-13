import argparse

import pandas as pd

import torch

# amm_resnet20, ...
from blink_mm.networks.third_party_resnet.amm_resnet_cifar import *
# amm_resnet18, ...
from blink_mm.networks.torchvision_resnet.amm_resnet import *
from blink_mm.count.main import counter, HOOKS
from blink_mm.expers.utils import factors
from blink_mm.expers.search.grid_search_cifar import _train


def dominate(a, b):
    return a["k"] >= b["k"] and a["subvec_len"] <= b["subvec_len"]


def find_skyline_by_brute_force(df, dominate):
    rows = df.to_dict(orient='index')
    skyline = set()
    for i in rows:
        dominated = False
        for j in rows:
            if i == j:
                continue
            if dominate(rows[j], rows[i]):
                dominated = True
                break
        if not dominated:
            skyline.add(i)

    return df.iloc[sorted(list(skyline)), :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train CIFAR")
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
    parser.add_argument("--model-type", default="amm_resnet18")
    parser.add_argument("--temperature-config", default="inverse")
    parser.add_argument("--temperature", nargs='*', type=float)
    parser.add_argument("--fix-weight", action="store_true")
    parser.add_argument("--replace-all", action="store_true")
    parser.add_argument("--flops-limit", default=138596352, type=int)

    args = parser.parse_args()

    df = pd.DataFrame()
    for k in range(8, 33):
        for subvec_len in sorted(list(factors(16 * 9))):
            model = globals()[args.model_type](
                k=k, subvec_len=subvec_len, cifar=True)
            dst = counter(model, (torch.randn(1, 3, 32, 32),), HOOKS)
            flops = dst["muladds"] + dst["adds"]
            if flops <= args.flops_limit:
                df = df.append({
                    "k": k,
                    "subvec_len": subvec_len
                }, ignore_index=True)

    df = find_skyline_by_brute_force(df, dominate)

    for _, row in df.iterrows():
        _train(args, int(row["k"]), int(row["subvec_len"]))
