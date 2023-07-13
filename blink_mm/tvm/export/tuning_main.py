import os
import argparse

import torch
import torch.utils.dlpack

import tvm
from tvm import autotvm, auto_scheduler, meta_schedule
import tvm.relay

from qat.export.export import _get_model_to_export

from blink_mm.tvm.export.utils import (
    quantize,
    tune_network,
    tune_network_auto_scheduler,
    tune_network_meta_schedule
)
from blink_mm.transforms.export.handlers import \
    AMMConv2dHandler, AMMLinearHandler, QuantizedAMMConv2dBatchNorm2dReLUHandler
from blink_mm.ops.amm_conv2d import AMMConv2d
from blink_mm.ops.amm_linear import AMMLinear
from blink_mm.ops.quant_amm_conv2d import QuantizedAMMConv2dBatchNorm2dReLU
from blink_mm.tvm.ops.amm_op_impl import amm_op_impl
from blink_mm.tvm.export.model_archive import MODEL_ARCHIVE


def tuning_main(
    num_threads, model, quantize_flag, tgt,
    tuner, host, port, key, tuning_records
):
    os.environ["TVM_NUM_THREADS"] = str(num_threads)

    model_info = MODEL_ARCHIVE[model]

    model = model_info["model"]()
    input_tensors = model_info["input"]
    model = _get_model_to_export(model, input_tensors, {
        AMMConv2d: AMMConv2dHandler,
        AMMLinear: AMMLinearHandler,
        QuantizedAMMConv2dBatchNorm2dReLU: QuantizedAMMConv2dBatchNorm2dReLUHandler,
    })
    scripted_model = torch.jit.trace(model, input_tensors).eval()

    input_infos = [
        (i.debugName().split('.')[0], i.type().sizes())
        for i in list(scripted_model.graph.inputs())[1:]
    ]
    custom_map = {"prim::PythonOp": amm_op_impl}
    mod, params = tvm.relay.frontend.from_pytorch(
        scripted_model, input_infos, custom_convert_map=custom_map)

    if quantize_flag:
        mod = quantize(mod, params, False)

    if tgt in ["x86", "x86_avx512"]:
        if tgt == "x86":
            target = "llvm -mcpu=core-avx2"
        elif tgt == "x86_avx512":
            target = "llvm -mcpu=cascadelake"
        if tuner == "autotvm":
            measure_option = autotvm.measure_option(
                builder="local", runner="local"
            )
        elif tuner == "auto_scheduler":
            builder = auto_scheduler.LocalBuilder()
            runner = auto_scheduler.LocalRunner(
                repeat=10, enable_cpu_cache_flush=True
            )
        elif tuner == "meta_schedule":
            builder = meta_schedule.builder.LocalBuilder()
            runner = meta_schedule.runner.LocalRunner()
            target += f" -num-cores={num_threads}"
    elif tgt == "arm":
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
        if tuner == "autotvm":
            measure_option = autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="ndk", timeout=60),
                runner=autotvm.RPCRunner(
                    key, host, port)
            )
        elif tuner == "auto_scheduler":
            builder = auto_scheduler.LocalBuilder(build_func="ndk", timeout=60)
            runner = auto_scheduler.RPCRunner(
                key,
                host=host,
                port=port,
                repeat=10,
                min_repeat_ms=200,
                enable_cpu_cache_flush=True,
            )
        elif tuner == "meta_schedule":
            builder = meta_schedule.builder.LocalBuilder(
                build_func="ndk", timeout=60)
            runner = meta_schedule.runner.RPCRunner(rpc_config=meta_schedule.runner.config.RPCConfig(
                tracker_host=host,
                tracker_port=port,
                tracker_key=key,
            ))
            target += f" -num-cores={num_threads}"

    if tuner == "autotvm":
        tuning_option = {
            "n_trial": 1500,
            "early_stopping": None,
            "measure_option": measure_option,
            "tuning_records": tuning_records,
        }

        tune_network(mod, params, target, tuning_option)
    elif tuner == "auto_scheduler":
        tuning_option = auto_scheduler.TuningOptions(
            num_measure_trials=15000,
            builder=builder,
            runner=runner,
            measure_callbacks=[
                auto_scheduler.RecordToFile(tuning_records)],
        )

        tune_network_auto_scheduler(mod, params, target, tuning_option)
    elif tuner == "meta_schedule":
        tuning_option = {
            "tune_config": meta_schedule.TuneConfig(
                max_trials_global=15000,
                num_trials_per_iter=64,
            ),
            "work_dir": tuning_records,
            "builder": builder,
            "runner": runner,
        }

        tune_network_meta_schedule(mod, params, target, tuning_option)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--tuning-records", default="resnet18.json")
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--tuner", default="autotvm",
                        choices=["autotvm", "auto_scheduler", "meta_schedule"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--target", default="x86",
                        choices=["x86", "x86_avx512", "arm"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9190, type=int)
    parser.add_argument("--key", default="pixel4")
    args = parser.parse_args()

    tuning_main(args.num_threads, args.model, args.quantize, args.target,
                args.tuner, args.host, args.port, args.key, args.tuning_records)
