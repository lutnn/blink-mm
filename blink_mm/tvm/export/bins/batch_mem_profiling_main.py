import subprocess
import argparse
import os.path as osp
import csv

from blink_mm.tvm.export.bins.config import batch_config
from adb_helper.adb import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="x86",
                        choices=["x86", "x86_avx512", "arm"])
    parser.add_argument("--model-bin-path")
    parser.add_argument("--dev-name")
    parser.add_argument("--serial")
    parser.add_argument("--csv-path")

    args = parser.parse_args()

    if args.target == "arm":
        adb = Adb(args.serial, False)
        adb.push("build/profile_tvm_model", "/data/local/tmp")
        adb.push("build/tvm_bin/libtvm_runtime.so", "/data/local/tmp")

    ls = batch_config(args.target)

    f = open(args.csv_path, "w")
    writer = csv.writer(f)
    writer.writerow(["model", "mem_mb"])

    for model, *_ in ls:
        lib_path = osp.join(args.model_bin_path,
                            args.dev_name, "1-threads", f"{model}.so")

        if args.target in ["x86", "x86_avx512"]:
            p = subprocess.Popen(
                ["build/profile_tvm_model", lib_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            mem_mb = float(p.communicate()[0].decode('utf-8')) / 1024
        elif args.target == "arm":
            adb.push(lib_path, "/data/local/tmp")
            output = adb.shell(
                f"LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/profile_tvm_model /data/local/tmp/{model}.so")
            lines = list(filter(
                lambda s: len(s.strip()) >= 1,
                output.split('\n')
            ))
            if len(lines) >= 2:
                print(lines)
            mem_mb = float(lines[-1]) / 1024

        writer.writerow([model, mem_mb])
        f.flush()

    f.close()
