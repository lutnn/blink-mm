from runner.hooks import Hook
from qat.ops import QuantizedOperator


class SwitchQuantizationModeHook(Hook):
    def __init__(self, switch_iter):
        super().__init__()
        self.switch_iter = switch_iter

    def after_train_iter(self, runner):
        if (runner.iter + 1) != self.switch_iter:
            return
        runner.logger.info("switching to activation quantization")
        for module in runner.model.modules():
            if isinstance(module, QuantizedOperator):
                module.activation_quantization = True
