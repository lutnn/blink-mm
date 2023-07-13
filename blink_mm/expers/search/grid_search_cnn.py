import os.path as osp
import argparse

from blink_mm.expers.train_cnn import (
    dist_train,
    get_device_id,
    dist_train_build,
    CheckpointHook,
    SwitchQuantizationModeHook,
    TensorboardLoggerHook,
    MonitorTemperatureHook,
    AnnealingTemperatureHook,
)
from blink_mm.expers.utils import factors


def _train(args, num_centroids, subvec_len):
    work_dir = osp.join(
        args.work_dir,
        f"k={num_centroids}-subvec_len={subvec_len}"
    )
    dist_train(
        args.num_procs, work_dir, args.num_epochs,
        get_device_id, dist_train_build, vars={
            "imgs_per_gpu": args.imgs_per_gpu,
            "root": args.root,
            "lr": args.lr,
            "temp_lr": args.temp_lr,
            "ckpt_path": args.ckpt_path,
            "model_type": args.model_type,
            "dataset_type": args.dataset_type,
            "device_ids": list(map(int, args.device_ids.split(','))),

            # ablation study
            "temperature_config": args.temperature_config,
            "temperature": args.temperature[0] if (args.temperature is not None and len(args.temperature) == 1) else None,
            "num_centroids": num_centroids,
            "subvec_len": subvec_len,
            "fix_weight": args.fix_weight,
            "replace_all": args.replace_all,
        },
        hooks=[
            CheckpointHook(save_best="accuracy"),
            SwitchQuantizationModeHook(5000),
            TensorboardLoggerHook(1),
            MonitorTemperatureHook(),
            *([AnnealingTemperatureHook(*args.temperature)]
              if args.temperature_config == "manual" else [])
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train CIFAR")
    parser.add_argument("--imgs-per-gpu", default=256, type=int)
    parser.add_argument("--root")
    parser.add_argument("--dataset-type", default="cifar10")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--temp-lr", default=1e-1, type=float)
    parser.add_argument("--num-epochs", default=50, type=int)
    parser.add_argument("--work-dir")
    parser.add_argument("--device-ids", default="0")
    parser.add_argument("--num-procs", default=1, type=int)
    parser.add_argument("--ckpt-path")
    parser.add_argument("--model-type", default="amm_resnet18_cifar")
    parser.add_argument("--temperature-config", default="inverse")
    parser.add_argument("--temperature", nargs='*', type=float)
    parser.add_argument("--fix-weight", action="store_true")
    parser.add_argument("--replace-all", action="store_true")

    args = parser.parse_args()

    for num_centroids in [8, 16, 32, 64]:
        for subvec_len_3x3 in [9, 18, 36]:
            _train(args, num_centroids, {
                "3x3": subvec_len_3x3,
                "1x1": 4
            })
