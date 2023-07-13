import torch

from .bert import BERT

from blink_mm.ops.amm_linear import AMMLinear
from qat.export.utils import replace_module_by_name, fetch_module_by_name


class AMMBERT(BERT):
    def __init__(
        self, num_labels,
        num_hidden_layers=12, num_layers_to_replace=6,
        k=16, subvec_len=32,
        name="bert-base-uncased", torchscript=False
    ):
        super().__init__(num_labels, name,
                         num_hidden_layers, torchscript=torchscript)
        self.num_hidden_layers = num_hidden_layers
        self._replace_with_amm_linear(
            num_layers_to_replace, k, subvec_len)

    def _replace_with_amm_linear(self, num_layers_to_replace, k, subvec_len):
        assert 768 % subvec_len == 0 and 3072 % subvec_len == 0

        ncodebooks = {
            "attention.self.query": 768 // subvec_len,
            "attention.self.key": 768 // subvec_len,
            "attention.self.value": 768 // subvec_len,
            "attention.output.dense": 768 // subvec_len,
            "intermediate.dense": 768 // subvec_len,
            "output.dense": 3072 // subvec_len,
        }

        for i in range(self.num_hidden_layers - num_layers_to_replace, self.num_hidden_layers):
            for name in ncodebooks:
                layer = self.transformer.bert.encoder.layer[i]
                module = fetch_module_by_name(layer, name)
                amm_linear = AMMLinear(
                    ncodebooks[name],
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    k=k
                )
                amm_linear.inverse_temperature_logit.data.copy_(
                    torch.tensor(10)
                )
                replace_module_by_name(layer, name, amm_linear)
