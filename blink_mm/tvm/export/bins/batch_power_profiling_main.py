import os
import os.path as osp
import argparse
import subprocess
import time

from adb_helper.adb import Adb, Android

from blink_mm.tvm.export.bins.config import batch_config


def profile_energy(adb, csv_name, seconds, dst_folder):
    # run energy_profiler for several seconds
    energy_profiler_proc = subprocess.Popen(
        ["adb", "-s", adb.adb_device_id, "shell", "su" if adb.su else ""],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    shell = f"/data/local/tmp/energy_profiler /data/local/tmp/{csv_name}"
    energy_profiler_proc.stdin.write(bytes(shell + "\n", "utf-8"))
    energy_profiler_proc.stdin.flush()
    time.sleep(seconds)
    energy_profiler_proc.stdin.write(bytes("stop\n", "utf-8"))
    energy_profiler_proc.stdin.flush()

    for name in ["profile_tvm_model", "energy_profiler"]:
        adb.shell(
            f"ps -A | grep {name} | awk '{{print $2}}' | xargs kill")
    os.makedirs(dst_folder, exist_ok=True)
    adb.pull(f"/data/local/tmp/{csv_name}", dst_folder)
    adb.shell(f"rm /data/local/tmp/{csv_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial")
    parser.add_argument("--bin-path", default="tvm-bin")
    parser.add_argument("--dev-name", default="pixel4")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--result-folder")
    args = parser.parse_args()

    result_folder = osp.join(osp.expanduser(args.result_folder), args.dev_name)

    adb = Adb(args.serial, True)
    android = Android(adb)

    adb.push("build/energy_profiler", "/data/local/tmp")
    adb.push("build/profile_tvm_model", "/data/local/tmp")
    adb.push("build/tvm_bin/libtvm_runtime.so", "/data/local/tmp")

    profile_energy(adb, "idle.csv", args.seconds, result_folder)

    ls = batch_config("arm")

    for model, *_ in ls:
        lib_path = osp.join(args.bin_path,
                            args.dev_name, "1-threads", f"{model}.so")

        adb.push(lib_path, "/data/local/tmp")

        # run profile_tvm_model
        profile_tvm_model_proc = subprocess.Popen(
            ["adb", "-s", adb.adb_device_id, "shell", "su" if adb.su else ""],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        shell = f"LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/profile_tvm_model /data/local/tmp/{model}.so inf"
        profile_tvm_model_proc.stdin.write(bytes(shell + "\n", "utf-8"))
        profile_tvm_model_proc.stdin.flush()

        time.sleep(1)
        output = adb.shell(
            "ps -A -o %cpu,%mem,name | grep profile_tvm_model")
        print(output)
        assert "profile_tvm_model" in output

        profile_energy(adb, f"{model}.csv", args.seconds, result_folder)
