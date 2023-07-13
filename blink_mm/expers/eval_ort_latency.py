import os
import os.path as osp
import argparse
import uuid
import subprocess

import torch

from adb_helper.adb import Adb, Android

from blink_mm.tvm.export.model_archive import MODEL_ARCHIVE


def evaluate_onnxruntime_for_linux(perf_test, model_folder, profile_folder, max_num_threads):
    os.makedirs(model_folder, exist_ok=True)

    for model_name in MODEL_ARCHIVE:
        if model_name.startswith("amm_"):
            continue
        model = MODEL_ARCHIVE[model_name]["model"]()
        model_input = MODEL_ARCHIVE[model_name]["input"]
        model_path = osp.join(model_folder, f"{model_name}.onnx")
        torch.onnx.export(model, model_input, model_path)

        num_threads = 1
        while num_threads <= max_num_threads:
            print(
                f"Evaluating {model_name} on ONNX Runtime with {num_threads} threads"
            )
            tmp_path = f"/tmp/{str(uuid.uuid1())}"
            p = subprocess.Popen(
                f"{perf_test} "
                f"-m times -I -e cpu -r 100 -x {num_threads} -y 1 "
                f"-p {tmp_path} "
                f"{model_path} ",
                stdout=subprocess.PIPE,
                shell=True
            )
            print(p.communicate()[0].decode('utf-8'))
            os.makedirs(
                f"{profile_folder}/{num_threads}-threads", exist_ok=True)
            os.system(
                f"mv {tmp_path}* {profile_folder}/{num_threads}-threads/{model_name}.json")
            num_threads *= 2


def evaluate_onnxruntime_for_android(adb: Adb, taskset, perf_test, model_folder, profile_folder, max_num_threads):
    android = Android(adb)
    if not android.boolean("-d /data/local/tmp/onnx_models"):
        adb.shell("mkdir /data/local/tmp/onnx_models")
    if android.boolean("-d /data/local/tmp/onnxruntime_profiles"):
        adb.shell("rm -r /data/local/tmp/onnxruntime_profiles")
    adb.shell("mkdir /data/local/tmp/onnxruntime_profiles")
    adb.push(perf_test, "/data/local/tmp")
    os.makedirs(model_folder, exist_ok=True)

    for model_name in MODEL_ARCHIVE:
        if model_name.startswith("amm_"):
            continue
        model = MODEL_ARCHIVE[model_name]["model"]()
        model_input = MODEL_ARCHIVE[model_name]["input"]
        model_path = osp.join(model_folder, f"{model_name}.onnx")
        torch.onnx.export(model, model_input, model_path)

        adb.push(model_path, "/data/local/tmp/onnx_models")

        num_threads = 1
        while num_threads <= max_num_threads:
            print(
                f"Evaluating {model_name} on ONNX Runtime with {num_threads} threads"
            )
            output = adb.shell(
                f"taskset {taskset} "
                f"/data/local/tmp/onnxruntime_perf_test "
                f"-m times -I -e cpu -r 100 -x {num_threads} -y 1 "
                f"-p /data/local/tmp/onnxruntime_profiles/{model_name} "
                f"/data/local/tmp/onnx_models/{model_name}.onnx "
            )
            print(output)
            os.makedirs(
                f"{profile_folder}/{num_threads}-threads", exist_ok=True)
            profile_file_path = adb.shell(
                f"ls /data/local/tmp/onnxruntime_profiles/{model_name}*").strip()
            adb.pull(
                profile_file_path,
                osp.join(profile_folder,
                         f"{num_threads}-threads/{model_name}.json")
            )
            adb.shell(f"rm {profile_file_path}")
            num_threads *= 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--ort-path", default="~/onnxruntime")
    parser.add_argument("--serial", default="")
    parser.add_argument("--taskset", default="80")
    parser.add_argument("--model-folder", default="./")
    parser.add_argument("--profile-folder")
    parser.add_argument("--dev-name")
    parser.add_argument("--max-num-threads", type=int)
    args = parser.parse_args()

    ort_path = osp.expanduser(args.ort_path)
    profile_folder = osp.join(
        osp.expanduser(args.profile_folder), args.dev_name)

    if args.serial != "":
        adb = Adb(args.serial, False)
        perf_test = osp.join(
            ort_path,
            "build/Android/Release/onnxruntime_perf_test"
        )
        evaluate_onnxruntime_for_android(
            adb, args.taskset, perf_test, args.model_folder, profile_folder, args.max_num_threads)
    else:
        perf_test = osp.join(
            ort_path,
            "build/Linux/Release/onnxruntime_perf_test"
        )
        evaluate_onnxruntime_for_linux(
            perf_test, args.model_folder, profile_folder, args.max_num_threads)
