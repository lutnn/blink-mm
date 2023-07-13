from math import sqrt
from functools import reduce

import torch

from tqdm import tqdm
import numpy as np


def get_params(model):
    temp_suffixes = [".inverse_temperature_logit", ".temperature"]
    centroids_suffixes = [".centroids"]
    base_params = [
        v for k, v in model.named_parameters()
        if np.all([
            not k.endswith(suffix)
            for suffix in temp_suffixes + centroids_suffixes
        ])
    ]
    centroids_params = [
        v for k, v in model.named_parameters()
        if np.any([k.endswith(suffix) for suffix in centroids_suffixes])
    ]
    temp_params = [
        v for k, v in model.named_parameters()
        if np.any([k.endswith(suffix) for suffix in temp_suffixes])
    ]
    return base_params, centroids_params, temp_params


def assign_temperature(temperature_config, temperature, temp_params):
    for param in temp_params:
        if temperature_config == "direct":
            param.data.copy_(torch.tensor(
                temperature, dtype=param.dtype, device=param.device))
        elif temperature_config == "inverse":
            # softplus(inverse_temperature_logit) + 1 = 1 / temperature
            if 1 / temperature >= 21:
                inverse_temperature_logit = 1 / temperature - 1
            else:
                inverse_temperature_logit = \
                    np.log(np.exp(1 / temperature - 1) - 1)
            param.data.copy_(torch.tensor(
                inverse_temperature_logit, dtype=param.dtype, device=param.device))


def evaluate_model(model, test_data_loader, model_name):
    model.eval()
    xs, ys = [], []
    for data_batch in tqdm(test_data_loader, desc=f"Evaluating {model_name}"):
        with torch.no_grad():
            x, y = model.val_step(model, data_batch, None)
        xs.append(x)
        ys.append(y)

    return model.evaluate(xs, ys)


def factors(n):
    step = 2 if n % 2 else 1
    return set(reduce(list.__add__,
                      ([i, n//i] for i in range(1, int(sqrt(n))+1, step) if n % i == 0)))
