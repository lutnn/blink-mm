from typing import Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import faiss
from tqdm import tqdm

from qat.ops import QuantizedConv2dBatchNorm2dReLU
from qat.export.utils import fetch_module_by_name
from blink_mm.im2col import im2col
from blink_mm.ops.maddness.maddness_conv2d import MaddnessConv2d
from blink_mm.ops.maddness.maddness_linear import MaddnessLinear
from blink_mm.ops.amm_conv2d import AMMConv2d
from blink_mm.ops.amm_linear import AMMLinear
from .utils import collect_input_tensors


def _train_pq(d, ncodebooks, num_centroids, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # data: (n, d)
    assert d % ncodebooks == 0
    assert len(data.shape) == 2 and data.shape[1] == d

    n, _ = data.shape
    subvec_len = d // ncodebooks
    centroids = np.empty(
        (ncodebooks, num_centroids, subvec_len), dtype=np.float32)
    index = np.empty((ncodebooks, n), dtype=np.int64)
    for i in range(ncodebooks):
        train_data = data[:, i * subvec_len: (i + 1) * subvec_len]
        train_data = np.ascontiguousarray(train_data)
        kmeans = faiss.Kmeans(subvec_len, num_centroids, gpu=True)
        kmeans.train(train_data)
        index[i] = kmeans.assign(train_data)[1]
        centroids[i] = kmeans.centroids

    return centroids, index


def _sync_centroids(modules, centroids):
    if dist.is_initialized():
        dist.barrier()
        for module in modules:
            if dist.get_rank() == 0:
                rank = torch.tensor(len(centroids[module].shape))
            else:
                rank = torch.tensor(0)
            dist.broadcast(rank, 0)
            if dist.get_rank() == 0:
                shape = torch.tensor(centroids[module].shape)
            else:
                shape = torch.empty(rank.item()).to(torch.long)
            dist.broadcast(shape, 0)
            if dist.get_rank() > 0:
                centroids[module] = torch.empty(tuple(shape.numpy())) \
                    .to(torch.float32)
            dist.broadcast(centroids[module], 0)


class TransferHandler:
    def __init__(self, model, target_model, calibrate_dataset) -> None:
        ...

    def transfer_conv2d(self, conv2d: nn.Conv2d, target: nn.Conv2d):
        target.weight.data.copy_(conv2d.weight.data)
        if target.bias is not None:
            target.bias.data.copy_(conv2d.bias.data)

    def transfer_bn2d(self, bn2d: nn.BatchNorm2d, target: nn.BatchNorm2d):
        target.load_state_dict(bn2d.state_dict())

    def transfer_linear(self, linear: nn.Linear, target: nn.Linear):
        target.weight.data.copy_(linear.weight.data)
        if target.bias is not None:
            target.bias.data.copy_(linear.bias.data)

    def transfer(self, module, target):
        target.load_state_dict(module.state_dict())

    def transfer_conv2d_bn2d(
        self, conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d,
        target: QuantizedConv2dBatchNorm2dReLU
    ):
        target.load_state_dict(conv2d.state_dict())


class AMMConv2dTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset):
        super().__init__(model, target_model, calibrate_dataset)
        self.centroids = self._learn_centroids(
            model, target_model, calibrate_dataset
        )

    @staticmethod
    def _learn_centroids(model, target_model, calibrate_dataset):
        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        centroids = {}

        def is_amm_conv2d(m):
            if not isinstance(m, nn.Conv2d):
                return False
            target_module = fetch_module_by_name(target_model, module_names[m])
            return isinstance(target_module, AMMConv2d)

        def ncodebooks(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "ncodebooks")

        def num_centroids(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "k")

        if not dist.is_initialized() or dist.get_rank() == 0:
            input_tensors = collect_input_tensors(
                model, calibrate_dataset,
                is_amm_conv2d
            )
            for conv, input_tensor in tqdm(list(input_tensors.items()), desc="computing centroids"):
                cols = im2col(
                    input_tensor.data.cpu().numpy(),
                    conv.kernel_size, conv.stride, conv.padding
                )
                pq_centroids, _ = _train_pq(
                    cols.shape[1],
                    ncodebooks(conv),
                    num_centroids(conv),
                    np.transpose(cols, (0, 2, 1)).reshape((-1, cols.shape[1]))
                )
                centroids[conv] = torch.tensor(pq_centroids)

        modules = list(filter(is_amm_conv2d, module_names.keys()))
        _sync_centroids(modules, centroids)

        return centroids

    def transfer_conv2d_bn2d(
        self, conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d,
        target: AMMConv2d
    ):
        self.transfer(conv2d, target)

    def transfer(self, conv2d: nn.Conv2d, target: AMMConv2d):
        target.centroids.data.copy_(self.centroids[conv2d].data)
        target.weight.data.copy_(
            conv2d.weight.data.reshape(conv2d.out_channels, -1).permute(1, 0).reshape_as(target.weight))
        if target.bias is not None:
            target.bias.data.copy_(conv2d.bias.data)


class AMMLinearTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset):
        super().__init__(model, target_model, calibrate_dataset)
        self.centroids = self._learn_centroids(
            model, target_model, calibrate_dataset
        )

    @staticmethod
    def _learn_centroids(model, target_model, calibrate_dataset):
        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        centroids = {}

        def is_amm_linear(m):
            if not isinstance(m, nn.Linear):
                return False
            target_module = fetch_module_by_name(target_model, module_names[m])
            return isinstance(target_module, AMMLinear)

        def ncodebooks(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "ncodebooks")

        def num_centroids(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "k")

        if not dist.is_initialized() or dist.get_rank() == 0:
            input_tensors = collect_input_tensors(
                model, calibrate_dataset,
                is_amm_linear
            )
            for linear, input_tensor in tqdm(list(input_tensors.items()), desc="computing centroids"):
                pq_centroids, _ = _train_pq(
                    linear.in_features,
                    ncodebooks(linear),
                    num_centroids(linear),
                    input_tensor.flatten(0, -2).cpu().numpy()
                )
                centroids[linear] = torch.tensor(pq_centroids)

        modules = list(filter(is_amm_linear, module_names.keys()))
        _sync_centroids(modules, centroids)

        return centroids

    def transfer(self, linear: nn.Linear, target: AMMLinear):
        target.centroids.data.copy_(self.centroids[linear].data)
        target.weight.data.copy_(
            linear.weight.data.permute(1, 0).reshape_as(target.weight))
        if target.bias is not None:
            target.bias.data.copy_(linear.bias.data)


class MaddnessConv2dTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset) -> None:
        super().__init__(model, target_model, calibrate_dataset)

        self.dic = self._learn_dt(model, target_model, calibrate_dataset)

    @staticmethod
    def _learn_dt(model, target_model, calibrate_dataset):
        from blink_mm.ops.maddness.maddness import _learn_codebooks, _optimize_prototypes, _create_lookup_tables

        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        def is_maddness_conv2d(m):
            if not isinstance(m, nn.Conv2d):
                return False
            target_module = fetch_module_by_name(target_model, module_names[m])
            return isinstance(target_module, MaddnessConv2d)

        def ncodebooks(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "ncodebooks")

        dic = {
            "split_idxs": {},
            "split_vals": {},
            "lookup_tables": {},
        }

        if not dist.is_initialized() or dist.get_rank() == 0:
            input_tensors = collect_input_tensors(
                model, calibrate_dataset,
                is_maddness_conv2d
            )

            for conv, input_tensor in tqdm(list(input_tensors.items()), desc="computing decision trees"):
                x = im2col(
                    input_tensor.data.cpu().numpy(),
                    conv.kernel_size, conv.stride, conv.padding
                )
                x = np.transpose(x, (0, 2, 1)).reshape((-1, x.shape[1]))
                x = x + np.random.rand(*x.shape) * 1e-6

                codebooks = _learn_codebooks(ncodebooks(conv), x)
                codebooks = _optimize_prototypes(x, codebooks)
                lookup_tables = _create_lookup_tables(
                    conv.weight.reshape(conv.out_channels, -1)
                        .transpose(1, 0).detach().cpu().numpy(),
                    codebooks
                )
                split_idxs = torch.tensor([
                    codebook.split_idxs
                    for codebook in codebooks
                ])
                split_vals = torch.empty(
                    ncodebooks(conv), 4, 8, dtype=torch.float32)
                for i in range(ncodebooks(conv)):
                    for j in range(4):
                        split_vals[i][j][:1 << j] = \
                            torch.tensor(codebooks[i].split_vals[j])

                dic["split_idxs"][conv] = split_idxs.detach().cpu()
                dic["split_vals"][conv] = split_vals.detach().cpu()
                dic["lookup_tables"][conv] = \
                    torch.tensor(np.array(lookup_tables))

        modules = list(filter(is_maddness_conv2d, module_names.keys()))
        _sync_centroids(modules, dic["split_idxs"])
        _sync_centroids(modules, dic["split_vals"])
        _sync_centroids(modules, dic["lookup_tables"])

        return dic

    def transfer_conv2d_bn2d(
        self, conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d,
        target: MaddnessConv2d
    ):
        self.transfer(conv2d, target)

    def transfer(self, conv2d: nn.Conv2d, target: MaddnessConv2d):
        target.split_idxs.data.copy_(self.dic["split_idxs"][conv2d])
        target.split_vals.data.copy_(self.dic["split_vals"][conv2d])
        target.lookup_tables.data.copy_(self.dic["lookup_tables"][conv2d])
        if target.bias is not None:
            target.bias.data.copy_(conv2d.bias.data)


class MaddnessLinearTransferHandler(TransferHandler):
    def __init__(self, model, target_model, calibrate_dataset) -> None:
        super().__init__(model, target_model, calibrate_dataset)

        self.dic = self._learn_dt(model, target_model, calibrate_dataset)

    @staticmethod
    def _learn_dt(model, target_model, calibrate_dataset):
        from blink_mm.ops.maddness.maddness import _learn_codebooks, _optimize_prototypes, _create_lookup_tables

        module_names = OrderedDict([
            (module, name) for name, module in model.named_modules()
        ])

        def is_maddness_linear(m):
            if not isinstance(m, nn.Linear):
                return False
            target_module = fetch_module_by_name(target_model, module_names[m])
            return isinstance(target_module, MaddnessLinear)

        def ncodebooks(m):
            return getattr(fetch_module_by_name(target_model, module_names[m]), "ncodebooks")

        dic = {
            "split_idxs": {},
            "split_vals": {},
            "lookup_tables": {},
        }

        if not dist.is_initialized() or dist.get_rank() == 0:
            input_tensors = collect_input_tensors(
                model, calibrate_dataset,
                is_maddness_linear
            )

            for linear, input_tensor in tqdm(list(input_tensors.items()), desc="computing decision trees"):
                x = input_tensor.detach().cpu().numpy()
                x = x + np.random.rand(*x.shape) * 1e-6

                codebooks = _learn_codebooks(ncodebooks(linear), x)
                codebooks = _optimize_prototypes(x, codebooks)
                lookup_tables = _create_lookup_tables(
                    linear.weight.transpose(1, 0).detach().cpu().numpy(),
                    codebooks
                )
                split_idxs = torch.tensor([
                    codebook.split_idxs
                    for codebook in codebooks
                ])
                split_vals = torch.empty(
                    ncodebooks(linear), 4, 8, dtype=torch.float32)
                for i in range(ncodebooks(linear)):
                    for j in range(4):
                        split_vals[i][j][:1 << j] = \
                            torch.tensor(codebooks[i].split_vals[j])

                dic["split_idxs"][linear] = split_idxs.detach().cpu()
                dic["split_vals"][linear] = split_vals.detach().cpu()
                dic["lookup_tables"][linear] = \
                    torch.tensor(np.array(lookup_tables))

        modules = list(filter(is_maddness_linear, module_names.keys()))
        _sync_centroids(modules, dic["split_idxs"])
        _sync_centroids(modules, dic["split_vals"])
        _sync_centroids(modules, dic["lookup_tables"])

        return dic

    def transfer(self, linear: nn.Linear, target: MaddnessLinear):
        target.split_idxs.data.copy_(self.dic["split_idxs"][linear])
        target.split_vals.data.copy_(self.dic["split_vals"][linear])
        target.lookup_tables.data.copy_(self.dic["lookup_tables"][linear])
        if target.bias is not None:
            target.bias.data.copy_(linear.bias.data)
