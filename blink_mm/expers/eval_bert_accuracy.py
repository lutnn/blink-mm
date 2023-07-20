import os.path as osp
import argparse
from glob import glob
import csv

import torch

from blink_mm.data.glue import CONFIG


def fast_evaluate(ckpt_folder, dataset, lr, metric):
    folder = osp.join(ckpt_folder, dataset, f"lr={lr}")
    model_paths = glob(osp.join(folder, "*_best_*_epoch_*.pth"))
    assert len(model_paths) == 1
    model_path = model_paths[0]
    ckpt = torch.load(model_path, map_location="cpu")
    return ckpt["meta"]["validation"][metric]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--ckpt-folder")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    ckpt_folder = osp.expanduser(args.ckpt_folder)

    f = open(args.output_csv, "w")
    writer = csv.writer(f)
    writer.writerow(["model", "number", "dataset", "metric"])

    for dataset in ["sst2", "qqp", "qnli", "rte"]:
        metric = CONFIG["metrics"][dataset]
        max_number = float("-inf")
        for lr in [2e-5, 3e-5, 4e-5, 5e-5]:
            print(f"Evaluating LUT-NN BERT {metric} for {dataset} dataset with lr={lr}")
            number = fast_evaluate(ckpt_folder, dataset, lr, metric)
            max_number = max(max_number, number)
        writer.writerow(["amm_bert", max_number, dataset, metric])
        f.flush()

    f.close()
