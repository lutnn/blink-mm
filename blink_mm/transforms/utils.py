from typing import Callable, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def collect_input_tensors(
    model: nn.Module, dataset: Dataset,
    func: Callable[[nn.Module], bool],
    batch_size=8,
    is_input_tensor=True,
):
    if len(dataset) == 0:
        return {}

    dst = OrderedDict({})

    def hook(module, input_tensor, output_tensor):
        if is_input_tensor:
            assert len(input_tensor) == 1
            tensor = input_tensor[0].cpu()
        else:
            tensor = output_tensor.cpu()
        if dst.get(module, None) is None:
            dst[module] = [tensor]
        else:
            dst[module].append(tensor)

    model.eval()
    handles = []
    dst.clear()
    for _, module in model.named_modules():
        if func(module):
            handles.append(module.register_forward_hook(hook))
    if len(handles) == 0:
        return dst
    train_data_loader = DataLoader(dataset, batch_size, shuffle=False)
    desc = f"collecting {'input' if is_input_tensor else 'output'} tensors"
    for data_batch in tqdm(train_data_loader, desc=desc):
        with torch.no_grad():
            model.val_step(model, data_batch, None)
    for handle in handles:
        handle.remove()

    for key in dst:
        with torch.no_grad():
            dst[key] = torch.concat(dst[key])

    return dst
