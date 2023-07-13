import os
import os.path as osp
import argparse

from blink_mm.tvm.export.tuning_main import tuning_main
from blink_mm.tvm.export.bins.config import batch_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--tuning-records")
    parser.add_argument("--target", default="arm",
                        choices=["arm", "x86", "x86_avx512"])
    parser.add_argument("--dev-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9190, type=int)
    parser.add_argument("--key", default="pixel4")

    args = parser.parse_args()

    ls = batch_config(args.target)

    for model, tuner, filename, _, tune_for_different_thread_number in ls:
        if not tune_for_different_thread_number and args.num_threads >= 2:
            continue

        tuning_records = osp.join(
            args.tuning_records, tuner, args.dev_name,
            f"{args.num_threads}-threads",
            filename
        )
        os.makedirs(osp.dirname(tuning_records), exist_ok=True)

        tuning_main(
            args.num_threads, model, False, args.target,
            tuner, args.host, args.port, args.key, tuning_records
        )
