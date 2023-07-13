import os

import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ORT, LUT-NN, TVM

TABLE_MAPPING = {
    "pixel6": {
        "vgg11_cifar": [
            # ["/features/features.0/Conv", 1, 1],
            ["/features/features.3/Conv", 3, 3],
            ["/features/features.7/Conv", 5, 5],
            ["/features/features.10/Conv", 6, 6],
            ["/features/features.14/Conv", 8, 8],
            ["/features/features.17/Conv", 9, 9],
            ["/features/features.21/Conv", 11, 11],
            ["/features/features.24/Conv", 12, 12],
            # ["/classifier/Gemm", 16, 16],
        ],
        "vgg11_bn": [
            # ["/features/features.0/Conv", 1, 1],
            ["/features/features.4/Conv", 4, 4],
            ["/features/features.8/Conv", 6, 6],
            ["/features/features.11/Conv", 7, 7],
            ["/features/features.15/Conv", 9, 9],
            ["/features/features.18/Conv", 10, 10],
            ["/features/features.22/Conv", 12, 12],
            ["/features/features.25/Conv", 13, 13],
            # ["/classifier/Gemm", 17, 17],
        ],
        "bert_last_6_layers": {
            "filenames": [
                "bert_last_6_layers.json",
                "amm_bert_for_layerwise_benchmark.csv",
                "bert_last_6_layers.csv",
            ],
            "ops": [
                ["/encoder/layer.0/attention/self/key/MatMul", 0, 1],
                ["/encoder/layer.0/attention/self/query/MatMul", 0, 3],
                ["/encoder/layer.0/attention/self/value/MatMul", 0, 9],
                ["/encoder/layer.0/attention/output/dense/MatMul", 0, 13],
                ["/encoder/layer.0/intermediate/dense/MatMul", 1, 19],
                ["/encoder/layer.0/output/dense/MatMul", 2, 21]
            ]}
    },
    "x86_server": {
        "vgg11_cifar": [
            # ["/features/features.2/Relu_output_0_nchwc", 1, 1],
            ["/features/features.5/Relu_output_0_nchwc", 3, 3],
            ["/features/features.9/Relu_output_0_nchwc", 5, 6],
            ["/features/features.12/Relu_output_0_nchwc", 6, 8],
            ["/features/features.16/Relu_output_0_nchwc", 8, 11],
            ["/features/features.19/Relu_output_0_nchwc", 9, 13],
            ["/features/features.23/Relu_output_0_nchwc", 11, 16],
            ["/features/features.26/Relu_output_0_nchwc", 12, 18],
            # ["/classifier/Gemm", 16, 22],
        ],
        "vgg11_bn": [
            # ["/features/features.2/Relu_output_0_nchwc", 1, 1],
            ["/features/features.6/Relu_output_0_nchwc", 4, 3],
            ["/features/features.10/Relu_output_0_nchwc", 6, 6],
            ["/features/features.13/Relu_output_0_nchwc", 7, 8],
            ["/features/features.17/Relu_output_0_nchwc", 9, 11],
            ["/features/features.20/Relu_output_0_nchwc", 10, 13],
            ["/features/features.24/Relu_output_0_nchwc", 12, 16],
            ["/features/features.27/Relu_output_0_nchwc", 13, 18],
            # ["/classifier/Gemm", 17, 22],
        ],
        "bert_last_6_layers": {
            "filenames": [
                "bert_last_6_layers.json",
                "amm_bert_for_layerwise_benchmark.csv",
                "bert_last_6_layers.csv",
            ],
            "ops": [
                ["/encoder/layer.0/attention/self/key/MatMul", 0, 1],
                ["/encoder/layer.0/attention/self/query/MatMul", 0, 3],
                ["/encoder/layer.0/attention/self/value/MatMul", 0, 9],
                ["/encoder/layer.0/attention/output/dense/MatMul", 0, 13],
                ["/encoder/layer.0/intermediate/dense/MatMul", 1, 19],
                ["/encoder/layer.0/output/dense/MatMul", 2, 21]
            ]
        }
    }
}

MODEL_NAMES = {
    "vgg11_cifar": "VGG11 (CIFAR10)",
    "vgg11_bn": "VGG11",
    "bert_last_6_layers": "BERT",
}

HARDWARE_NAMES = {
    "pixel6": "Pixel6",
    "x86_server": "x86 Server",
}


def ort_latency(obj, name):
    ls = []
    for i in obj:
        if i["name"] == f"{name}_kernel_time":
            ls.append(i["dur"])
    assert len(ls) >= 1
    if len(ls) > 10:
        ls = sorted(ls)[5:-5]
    return np.mean(ls) / 1e3


def tvm_latency(df: pd.DataFrame, row):
    return df.iloc[row]["Duration (us)"] / 1e3


def plot_ax(model_alias, hardware_alias, ax):
    title = f"{MODEL_NAMES[model_alias]}\non {HARDWARE_NAMES[hardware_alias]}"

    ls = TABLE_MAPPING[hardware_alias][model_alias]
    if isinstance(ls, dict):
        ort_filename, amm_filename, tvm_filename = ls["filenames"]
        ls = ls["ops"]
        with open(f"./ae-output/layerwise-latency-report/{hardware_alias}/1-threads/{ort_filename}", "r") as f:
            obj = json.load(f)
        amm_df = pd.read_csv(
            f"./ae-output/layerwise-latency-report/{hardware_alias}/1-threads/{amm_filename}")
        tvm_df = pd.read_csv(
            f"./ae-output/layerwise-latency-report/{hardware_alias}/1-threads/{tvm_filename}")
    else:
        with open(f"./ae-output/layerwise-latency-report/{hardware_alias}/1-threads/{model_alias}.json", "r") as f:
            obj = json.load(f)
        amm_df = pd.read_csv(
            f"./ae-output/layerwise-latency-report/{hardware_alias}/1-threads/amm_{model_alias}.csv")
        tvm_df = pd.read_csv(
            f"./ae-output/layerwise-latency-report/{hardware_alias}/1-threads/{model_alias}.csv")

    x = np.arange(len(ls))
    ort_latencies = np.array([
        ort_latency(obj, ls[i][0])
        for i in x
    ])
    amm_latencies = np.array([
        tvm_latency(amm_df, ls[i][1])
        for i in x
    ])
    tvm_latencies = np.array([
        tvm_latency(tvm_df, ls[i][2])
        for i in x
    ])

    over_ort = ort_latencies / amm_latencies
    over_tvm = tvm_latencies / amm_latencies

    ax.plot(x, over_ort, '--o', label="Speedup Over ORT")
    ax.plot(x, over_tvm, '--^', label="Speedup Over TVM")
    ax.set_title(title, fontsize=36)
    ax.yaxis.set_tick_params(labelsize=36)
    ax.set_xticks([])
    ax.locator_params(axis="y", nbins=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    pd.DataFrame({
        "tvm_shape": [tvm_df.iloc[ls[i][2]]["Argument Shapes"] for i in x],
        "ort_name": [ls[i][0] for i in x],
        "amm_latency": amm_latencies.tolist(),
        "ort_latency": ort_latencies.tolist(),
        "tvm_latency": tvm_latencies.tolist(),
    }).to_csv(f"./ae-output/layerwise-latency-report/{hardware_alias}/1-threads/{model_alias}-organized.csv", index=False)

    return pd.DataFrame({
        "model": [model_alias],
        "hardware": [hardware_alias],
        "min_speedup_over_ort": [over_ort.min()],
        "max_speedup_over_ort": [over_ort.max()],
        "min_speedup_over_tvm": [over_tvm.min()],
        "max_speedup_over_tvm": [over_tvm.max()],
    })


def plot_layerwise_latency():
    hardware_aliases = ["pixel6", "x86_server"]
    model_aliases = ["vgg11_cifar", "vgg11_bn", "bert_last_6_layers"]
    fig, axs = plt.subplots(
        nrows=len(model_aliases),
        ncols=len(hardware_aliases),
        figsize=(len(hardware_aliases) * 6, len(model_aliases) * 4.4))

    dfs = []

    for i, hardware_alias in enumerate(hardware_aliases):
        for j, model_alias in enumerate(model_aliases):
            df = plot_ax(model_alias, hardware_alias, axs[j, i])
            dfs.append(df)

    axs[0][0].legend(fontsize=36, bbox_to_anchor=(0, 2.45),
                     loc="upper left")

    fig.supylabel("Speedup", fontsize=36)
    fig.supxlabel("Kernel Index", fontsize=36)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.75, bottom=0.075,
                        wspace=0.4, hspace=0.5)
    plt.savefig("./ae-output/layerwise_latency.pdf")

    df = pd.concat(dfs)
    df.to_csv("./ae-output/layerwise_speedup.csv", index=False)


if __name__ == "__main__":
    plot_layerwise_latency()
