import csv

import pandas as pd

from blink_mm.tvm.export.model_archive import MODEL_ARCHIVE


def estimate_avg_power(filename):
    df = pd.read_csv(filename)
    avg_power_w = df.loc[1:, "power_w"].mean()
    return avg_power_w


if __name__ == "__main__":
    models = list(MODEL_ARCHIVE.keys())
    models.remove("amm_bert_for_layerwise_benchmark")

    f = open("ae-output/power-organized.csv", "w")
    writer = csv.writer(f)
    writer.writerow(["model", "avg_power_w"])

    idle_power_w = estimate_avg_power("ae-output/energy/pixel4/idle.csv")
    for model_name in models:
        power_w = estimate_avg_power(
            f"ae-output/energy/pixel4/{model_name}.csv")
        writer.writerow([model_name, power_w - idle_power_w])
        f.flush()

    f.close()
