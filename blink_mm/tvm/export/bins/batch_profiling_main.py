import os
import os.path as osp
import argparse

from blink_mm.tvm.export.profiling_main import profiling_main
from blink_mm.tvm.export.bins.config import batch_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--report", default="report")
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--tuning-records")
    parser.add_argument("--bin-path")
    parser.add_argument("--target", default="arm",
                        choices=["arm", "x86", "x86_avx512"])
    parser.add_argument("--dev-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9190, type=int)
    parser.add_argument("--key", default="pixel4")

    args = parser.parse_args()

    ls = batch_config(args.target)

    for model, tuner, filename, csvname, tune_for_different_thread_number in ls:
        tuning_records = osp.join(
            args.tuning_records, tuner, args.dev_name,
            f"{args.num_threads if tune_for_different_thread_number else 1}-threads",
            filename
        )
        report = osp.join(
            args.report, args.dev_name,
            f"{args.num_threads}-threads",
            csvname
        )
        bin_path = None if (args.bin_path is None) else osp.join(
            args.bin_path, args.dev_name,
            f"{args.num_threads if tune_for_different_thread_number else 1}-threads",
            model + ".so"
        )
        if bin_path is not None:
            os.makedirs(osp.dirname(bin_path), exist_ok=True)
        profiling_main(
            args.num_threads, model, None,
            False, args.target, 3, tuner,
            tuning_records, args.host, args.port, args.key,
            report, False, None, bin_path
        )
