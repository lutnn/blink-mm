import numpy as np
import torch.distributed as dist

from runner.hooks import Hook


class MonitorTemperatureHook(Hook):
    def __init__(self):
        super().__init__()

    def _model(self, runner):
        return runner.model.module if dist.is_initialized() else runner.model

    @staticmethod
    def _softplus(input, beta=1, threshold=20):
        if input * beta >= threshold:
            return input
        return 1 / beta * np.log(1 + np.exp(beta * input))

    def _get_temperature(self, model):
        temperatures = []
        for module in model.modules():
            if getattr(module, "inverse_temperature_logit", None) is not None:
                temperatures.append(
                    1 / (self._softplus(module.inverse_temperature_logit.item()) + 1)
                )
            elif getattr(module, "temperature", None) is not None:
                temperatures.append(max(1e-6, module.temperature.item()))
        return temperatures

    def after_train_iter(self, runner):
        model = self._model(runner)
        temperatures = self._get_temperature(model)
        if len(temperatures) >= 1:
            runner.log_buffer.output["temperature/mean"] = \
                np.mean(temperatures)
            runner.log_buffer.output["temperature/var"] = \
                np.var(temperatures)
