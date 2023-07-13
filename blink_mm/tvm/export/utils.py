import io

import pandas as pd
import numpy as np

from tvm import autotvm, relay, auto_scheduler, meta_schedule


def quantize(mod, params, data_aware, **kwargs):
    qconfig_kwargs = {
        "skip_dense_layer": False,
        "skip_conv_layers": []
    }
    if data_aware:
        with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max", **qconfig_kwargs):
            mod = relay.quantize.quantize(
                mod, params, dataset=kwargs["calibrate_dataset"]())
    else:
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0, **qconfig_kwargs):
            mod = relay.quantize.quantize(mod, params)
    return mod


def tune_network(mod, params, target, tuning_option):
    from tvm.autotvm.tuner import XGBTuner

    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params)

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d: %s] " % (i + 1, len(tasks), task.name)
        tuner = XGBTuner(task, loss_type="rank", feature_type="curve")
        tuner.tune(
            n_trial=min(tuning_option["n_trial"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(
                    tuning_option["n_trial"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )


def tune_network_auto_scheduler(mod, params, target, tuning_option):
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"], params, target)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tuning_option)


def tune_network_meta_schedule(mod, params, target, tuning_option):
    tasks = meta_schedule.extract_task_from_relay(
        mod["main"], target=target, params=params)
    meta_schedule.tune_extracted_tasks(
        tasks,
        config=tuning_option["tune_config"],
        work_dir=tuning_option["work_dir"],
        builder=tuning_option["builder"],
        runner=tuning_option["runner"],
    )


def merge_tvm_profiles(csvs):
    dfs = [pd.read_csv(io.StringIO(csv)) for csv in csvs]
    latency_column = "Duration (us)"
    columns = dfs[0].columns.values.tolist()
    columns.remove(latency_column)

    mean = np.zeros((len(dfs[0]), ))

    for df in dfs:
        mean += np.array(df[latency_column])

    mean /= len(dfs)
    dic = {
        column: dfs[0][column]
        for column in columns
    }
    dic[latency_column] = mean

    return pd.DataFrame(dic)
