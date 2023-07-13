import itertools
from typing import Any, Dict

from torch.utils.data import DataLoader


class DataReader:
    def __init__(self, input_names: Dict[str, Any], data_loader: DataLoader, num_iters):
        self.data_loader = data_loader
        self.num_iters = num_iters
        self.input_names = input_names
        self.to_numpy = self._to_numpy()

    def _to_numpy(self):
        for data_batch in itertools.islice(self.data_loader, self.num_iters):
            yield {
                key: data_batch[value].detach().cpu().numpy()
                for key, value in self.input_names.items()
            }

    def get_next(self):
        return next(self.to_numpy, None)
