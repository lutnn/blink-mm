import numpy as np
import torch
import torch.distributed as dist

from runner.hooks import Hook


class AnnealingTemperatureHook(Hook):
    def __init__(self, begin, end):
        super().__init__()
        self.begin = begin
        self.end = end

    def _model(self, runner):
        return runner.model.module if dist.is_initialized() else runner.model

    def _calc_temperature(self, progress: float):
        t = np.exp(
            (np.log(self.end) - np.log(self.begin)) * progress
            + np.log(self.begin)
        )
        return t

    def _assign_temperature(self, model, t):
        for module in model.modules():
            if getattr(module, "temperature", None) is not None:
                module.temperature.data.copy_(torch.tensor(t))

    def before_run(self, runner):
        model = self._model(runner)
        self._assign_temperature(model, self.begin)

    def after_train_iter(self, runner):
        model = self._model(runner)
        self._assign_temperature(model, self._calc_temperature(
            (runner.iter + 1) / runner.num_iters
        ))
