import os
import argparse
import shutil

import torch
import torch.utils.dlpack

import tvm
import tvm.testing
import tvm.relay
from tvm import rpc, autotvm, auto_scheduler, meta_schedule
from tvm.contrib import utils, ndk
import tvm.contrib.debugger.debug_executor as runtime

from qat.export.export import _get_model_to_export
from qat.ops import QuantizedTensor

from blink_mm.tvm.export.utils import merge_tvm_profiles, quantize
from blink_mm.transforms.export.handlers import \
    AMMConv2dHandler, AMMLinearHandler, QuantizedAMMConv2dBatchNorm2dReLUHandler
from blink_mm.ops.amm_conv2d import AMMConv2d
from blink_mm.ops.amm_linear import AMMLinear
from blink_mm.ops.quant_amm_conv2d import QuantizedAMMConv2dBatchNorm2dReLU
from blink_mm.tvm.ops.amm_op_impl import amm_op_impl
from blink_mm.tvm.export.model_archive import MODEL_ARCHIVE


def profiling_main(
    num_threads, model, ckpt_path,
    quantize_flag, tgt, opt_level, tuner,
    tuning_records, host, port, key, report_path,
    verify, code_path, bin_path
):
    print(f"Evaluating {model} with {num_threads} threads")
    os.environ["TVM_NUM_THREADS"] = str(num_threads)

    model_info = MODEL_ARCHIVE[model]

    model_params = {}
    if ckpt_path is not None:
        model_params["ckpt_path"] = ckpt_path
    model = model_info["model"](**model_params)
    input_data = model_info["input"]

    output_torch = model(*input_data)

    model = _get_model_to_export(model, input_data, {
        AMMConv2d: AMMConv2dHandler,
        AMMLinear: AMMLinearHandler,
        QuantizedAMMConv2dBatchNorm2dReLU: QuantizedAMMConv2dBatchNorm2dReLUHandler,
    })

    scripted_model = torch.jit.trace(model, input_data)
    scripted_model.eval()

    input_infos = [
        (i.debugName().split('.')[0], i.type().sizes())
        for i in list(scripted_model.graph.inputs())[1:]
    ]
    custom_map = {"prim::PythonOp": amm_op_impl}

    mod_net, params_net = tvm.relay.frontend.from_pytorch(
        scripted_model, input_infos, custom_convert_map=custom_map)

    if quantize_flag:
        mod_net = quantize(mod_net, params_net, False)

    if tgt == "x86":
        target = "llvm -mcpu=core-avx2"
    elif tgt == "x86_avx512":
        target = "llvm -mcpu=cascadelake"
    elif tgt == "arm":
        target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
    if tuner == "meta_schedule":
        target += f" -num-cores={num_threads}"

    if code_path is not None:
        with open(code_path + ".tir", "w") as file:
            ...  # delete the original file

    @tvm.tir.transform.prim_func_pass(opt_level=0)
    def print_tir(f, mod, ctx):
        if code_path is not None:
            with open(code_path + ".tir", "a") as file:
                file.write(str(f))
        return f

    def relay_build(use_auto_scheduler, use_meta_schedule):
        with tvm.transform.PassContext(
            opt_level=opt_level,
            config={
                "relay.backend.use_auto_scheduler": use_auto_scheduler,
                "relay.backend.use_meta_schedule": use_meta_schedule,
                "tir.add_lower_pass": [(3, print_tir)]
            },
            required_pass=["FastMath"]
        ):
            return tvm.relay.build(mod_net, target=target, params=params_net)

    if tuning_records is None:
        lib = relay_build(False, False)
    elif tuner == "autotvm":
        with autotvm.apply_history_best(tuning_records):
            lib = relay_build(False, False)
    elif tuner == "auto_scheduler":
        with auto_scheduler.ApplyHistoryBest(tuning_records):
            lib = relay_build(True, False)
    elif tuner == "meta_schedule":
        with meta_schedule.ApplyHistoryBest(meta_schedule.database.JSONDatabase(
            f"{tuning_records}/database_workload.json",
            f"{tuning_records}/database_tuning_record.json"
        )):
            lib = relay_build(False, True)

    if tgt in ["x86", "x86_avx512"]:
        ctx = tvm.runtime.device("cpu")
        m = runtime.create(lib.get_graph_json(), lib.get_lib(), ctx)
        if bin_path is not None:
            lib.export_library(bin_path)
    elif tgt == "arm":
        remote = rpc.connect_tracker(host, port).request(key)
        temp = utils.tempdir()
        dso_binary = "dev_lib.so"
        dso_binary_path = temp.relpath(dso_binary)
        ctx = remote.cpu(0)
        lib.export_library(dso_binary_path, ndk.create_shared)
        if bin_path is not None:
            shutil.copy(dso_binary_path, bin_path)
        remote.upload(dso_binary_path)
        rlib = remote.load_module(dso_binary)
        m = runtime.create(lib.get_graph_json(), rlib, ctx)

    m.set_input(**lib.get_params())
    for i, tensor in enumerate(input_data):
        m.set_input(input_infos[i][0], tvm.nd.array(tensor, ctx))

    print(m.benchmark(ctx, repeat=5, number=20))

    if report_path is not None:
        reports = []
        for i in range(100):
            reports.append(m.profile().csv())
        report = merge_tvm_profiles(reports)

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        report.to_csv(report_path, index=False)

    if verify:
        if "Tensor" in output_torch.__class__.__name__:
            output_torch = (output_torch,)

        for i in range(len(output_torch)):
            y = m.get_output(i).numpy()
            if isinstance(output_torch[i], QuantizedTensor):
                x = output_torch[i].q.numpy()
            else:
                x = output_torch[i].detach().numpy()

            tvm.testing.assert_allclose(x, y, rtol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--model", default="amm_resnet18")
    parser.add_argument("--report", default="report.csv")
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--ckpt-path", nargs="?")
    parser.add_argument("--tuner", default="autotvm",
                        choices=["autotvm", "auto_scheduler", "meta_schedule"])
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--tuning-records", nargs='?')
    parser.add_argument("--target", default="arm",
                        choices=["arm", "x86", "x86_avx512"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9190, type=int)
    parser.add_argument("--key", default="pixel4")
    parser.add_argument("--opt-level", default=3, type=int)
    parser.add_argument("--code-path", nargs="?")
    parser.add_argument("--bin-path", nargs="?")

    args = parser.parse_args()

    profiling_main(
        args.num_threads, args.model, args.ckpt_path, args.quantize,
        args.target, args.opt_level, args.tuner, args.tuning_records,
        args.host, args.port, args.key, args.report, True,
        args.code_path, args.bin_path
    )
