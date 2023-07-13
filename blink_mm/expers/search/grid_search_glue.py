import argparse
import os.path as osp

from runner.train import dist_train
from runner.hooks import TensorboardLoggerHook, CheckpointHook

from blink_mm.expers.train_glue import dist_train_build, get_device_id
from blink_mm.hooks.switch_quantization_mode_hook import SwitchQuantizationModeHook
from blink_mm.hooks.monitor_temperature_hook import MonitorTemperatureHook
from blink_mm.data.glue import CONFIG


def _train(args, k, subvec_len, lr, num_layers_to_replace):
    vars = {
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "lr": lr,
        "temp_lr": args.temp_lr,
        "dataset_type": args.dataset_type,
        "device_ids": list(map(int, args.device_ids.split(','))),
        "model_type": "amm_bert",
        "ckpt_path": osp.join(
            args.ckpt_dir, args.dataset_type, "epoch_3.pth"),
        "k": k,
        "subvec_len": subvec_len,
        "num_layers_to_replace": num_layers_to_replace,
    }
    work_dir = osp.join(
        args.work_dir,
        f"num_layers_to_replace={num_layers_to_replace}-k={k}-subvec_len={subvec_len}",
        f"lr={lr}"
    )
    dist_train(
        args.num_procs,
        work_dir,
        3,
        get_device_id, dist_train_build, vars=vars,
        hooks=[
            CheckpointHook(
                save_best=CONFIG["metrics"][args.dataset_type]),
            SwitchQuantizationModeHook(5000),
            TensorboardLoggerHook(1),
            MonitorTemperatureHook()
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grid Search GLUE")
    parser.add_argument("--batch-size-per-gpu", default=8, type=int)
    parser.add_argument("--temp-lr", default=1e-1, type=float)
    parser.add_argument("--work-dir")
    parser.add_argument("--device-ids", default="0,1,2,3")
    parser.add_argument("--num-procs", default=4, type=int)
    parser.add_argument("--ckpt-dir")
    parser.add_argument("--dataset-type", default="mnli")
    parser.add_argument("--grid-search-type",
                        default="layers", choices=["layers", "hyperparams"])

    args = parser.parse_args()

    if args.grid_search_type == "hyperparams":
        for k in [8, 16, 32, 64]:
            for subvec_len in [8, 16, 32, 64]:
                for lr in [3e-05]:
                    _train(args, k, subvec_len, lr, 6)
    elif args.grid_search_type == "layers":
        for num_layers_to_replace in [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
            _train(args, 16, 32, 3e-5, num_layers_to_replace)
