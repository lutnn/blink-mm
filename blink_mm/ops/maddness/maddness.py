import pickle
import logging
from typing import List, Tuple
from collections import namedtuple
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from runner.utils import config_logger

matplotlib.use('Agg')


CodeBook = namedtuple("CodeBook", ["split_idxs", "split_vals", "prototypes"])
# split_idxs: List[int]
# split_vals: List[List[float]]
# prototypes: np.ndarray, 16 * D or 16 * D/C


def _heuristic_select_idxs(buckets: List[np.ndarray]) -> List[int]:
    _, d = buckets[0].shape
    losses = [0 for _ in range(d)]
    for i in range(d):
        for bucket in buckets:
            losses[i] += np.sum((bucket[:, i] - np.mean(bucket[:, i])) ** 2)
    losses = sorted(list(zip(losses, range(d))), reverse=True)
    losses = losses[:4]
    return [t[1] for t in losses]


def _cumulative_sse(bucket, reverse: bool):
    n, d = bucket.shape
    iter = range(n)
    if reverse:
        iter = reversed(iter)
    x_sum = np.zeros((d,), dtype=np.float32)
    x_sq_sum = np.zeros((d,), dtype=np.float32)
    ret = np.zeros((n,), dtype=np.float32)
    tot = 0
    for i in iter:
        tot += 1
        x_sum += bucket[i]
        x_sq_sum += bucket[i] ** 2
        ret[i] = np.sum(x_sq_sum - (x_sum ** 2) / tot)
    return ret


def _optimal_split_threshold(idx, bucket) -> Tuple[float, float]:
    bucket = bucket[bucket[:, idx].argsort()]
    n, _ = bucket.shape
    assert n >= 2

    sses_head = _cumulative_sse(bucket, False)
    sses_tail = _cumulative_sse(bucket, True)

    best_loss = np.inf
    min_threshold = 0.1
    max_threshold = 0.9
    for i in range(int(min_threshold * n), min(int(max_threshold * n), n - 1)):
        # 0 .. i, i+1 .. n-1
        loss = sses_head[i] + sses_tail[i + 1]
        if bucket[i, idx] < bucket[i + 1, idx] and loss < best_loss:
            best_loss = loss
            best_val = (bucket[i, idx] + bucket[i + 1, idx]) / 2

    return best_val, best_loss


def _heuristic_split_threshold(idx, bucket) -> Tuple[float, float]:
    bucket = bucket[bucket[:, idx].argsort()]
    n, _ = bucket.shape
    assert n >= 2

    sses_head = _cumulative_sse(bucket, False)
    sses_tail = _cumulative_sse(bucket, True)

    def test(i):
        if bucket[i, idx] < bucket[i + 1, idx]:
            return True, (bucket[i, idx] + bucket[i + 1, idx]) / 2, sses_head[i] + sses_tail[i + 1]
        return False, None, None

    for i in range(n // 2 + 1):
        if n // 2 + i <= n - 2:
            flag, val, loss = test(n // 2 + i)
            if flag:
                return val, loss
        flag, val, loss = test(n // 2 - i)
        if flag:
            return val, loss

    return bucket[0, idx], np.inf


def _learn_hash_tree(subspace: np.ndarray) -> CodeBook:
    split_idxs = []
    split_vals = []

    buckets = [subspace]
    for _ in range(4):
        idxs = _heuristic_select_idxs(buckets)
        best_loss_sum = np.inf
        for j in idxs:
            loss_sum = 0
            vals = []
            for bucket in buckets:
                val, loss = _heuristic_split_threshold(j, bucket)
                vals.append(val)
                loss_sum += loss
            if loss_sum < best_loss_sum:
                best_loss_sum = loss_sum
                best_vals = vals
                best_idx = j

        split_idxs.append(best_idx)
        split_vals.append(best_vals)

        new_buckets = []
        for bucket, split_val in zip(buckets, best_vals):
            new_buckets.append(bucket[bucket[:, best_idx] < split_val])
            new_buckets.append(bucket[bucket[:, best_idx] >= split_val])
        buckets = new_buckets

    # the temporary prototypes are only for testing
    prototypes = np.vstack([np.mean(bucket, axis=0) for bucket in buckets])

    return CodeBook(split_idxs, split_vals, prototypes)


def _fast_learn_hash_tree(subspace: np.ndarray) -> CodeBook:
    split_idxs = []
    split_vals = []

    buckets = [subspace]
    for _ in range(4):
        idx = _heuristic_select_idxs(buckets)[0]
        vals = []
        for bucket in buckets:
            bucket = bucket[bucket[:, idx].argsort()]
            n = bucket.shape[0]
            val = (bucket[n // 2 - 1, idx] + bucket[n // 2, idx]) / 2
            vals.append(val)

        split_idxs.append(idx)
        split_vals.append(vals)

        new_buckets = []
        for bucket, split_val in zip(buckets, vals):
            new_buckets.append(bucket[bucket[:, idx] < split_val])
            new_buckets.append(bucket[bucket[:, idx] >= split_val])
        buckets = new_buckets

    # the temporary prototypes are only for testing
    prototypes = np.vstack([np.mean(bucket, axis=0) for bucket in buckets])

    return CodeBook(split_idxs, split_vals, prototypes)


def _encode(subspace: np.ndarray, codebook: CodeBook):
    n, _ = subspace.shape
    split_vals = codebook.split_vals
    split_idxs = codebook.split_idxs

    buckets = [np.arange(n)]
    for t in range(4):
        new_buckets = []
        for bucket, split_val in zip(buckets, split_vals[t]):
            new_buckets.append(
                bucket[subspace[bucket, split_idxs[t]] < split_val])
            new_buckets.append(
                bucket[subspace[bucket, split_idxs[t]] >= split_val])
        buckets = new_buckets

    # use uint8 since uint4 is not supported
    encoding = np.zeros((n,)).astype(np.uint8)
    for i, bucket in enumerate(buckets):
        encoding[bucket] = i
    return encoding


def _optimize_prototypes(a, codebooks: List[CodeBook]) -> List[CodeBook]:
    # after optimizing, the dimension of the prototypes become D
    # from the previous D/C

    ncodebooks = len(codebooks)
    assert ncodebooks & -ncodebooks == ncodebooks
    n, d = a.shape
    assert d % ncodebooks == 0
    subvec_len = d // ncodebooks

    encodings = [
        _encode(a[:, i * subvec_len: (i + 1) * subvec_len], codebooks[i])
        for i in range(ncodebooks)
    ]
    g = np.zeros((n, 16 * ncodebooks))
    for i, encoding in enumerate(encodings):
        g[np.arange(n), i * 16 + encoding] = 1

    prototypes = np.dot(np.dot(
        np.linalg.inv(np.dot(g.T, g) + np.eye(16 * ncodebooks)),
        g.T),
        a)
    for i in range(ncodebooks):
        codebooks[i] = CodeBook(
            codebooks[i].split_idxs, codebooks[i].split_vals,
            prototypes[i * 16: (i + 1) * 16, :]
        )

    return codebooks


def _create_lookup_tables(b, codebooks: List[CodeBook]) -> List[np.ndarray]:
    # lookup tables' shape: (16, M)
    lookup_tables = []
    for codebook in codebooks:
        lookup_tables.append(np.dot(codebook.prototypes, b))
    return lookup_tables


def _learn_codebooks(ncodebooks, a) -> List[CodeBook]:
    _, d = a.shape
    assert d % ncodebooks == 0
    subvec_len = d // ncodebooks

    codebooks = []
    for i in range(ncodebooks):
        codebook = _fast_learn_hash_tree(
            a[:, subvec_len * i: subvec_len * (i + 1)])
        codebooks.append(codebook)
    return codebooks


class Maddness:
    def __init__(self, ncodebooks, fit_add_bias):
        self.ncodebooks = ncodebooks
        self.fit_add_bias = fit_add_bias
        self.logger = logging.getLogger("maddness")

    def fit(self, a, b):
        _, self.d = a.shape
        if self.fit_add_bias:
            a = a + np.random.rand(*a.shape) * 1e-6
        _, self.m = b.shape
        assert self.d == b.shape[0]
        self.subvec_len = self.d // self.ncodebooks
        self.codebooks = _learn_codebooks(self.ncodebooks, a)
        self.codebooks = _optimize_prototypes(a, self.codebooks)
        self.lookup_tables = _create_lookup_tables(b, self.codebooks)

    def __call__(self, a) -> np.ndarray:
        n, d = a.shape
        assert self.d == d

        ret = np.zeros((n, self.m))

        for i in range(self.ncodebooks):
            encoding = _encode(
                a[:, self.subvec_len * i: self.subvec_len * (i + 1)],
                self.codebooks[i]
            )
            ret += self.lookup_tables[i][encoding]

        return ret


def _test_hashing():
    a = np.random.rand(512, 2)
    codebook = _learn_hash_tree(a)
    encoding = _encode(a, codebook)
    for i in range(16):
        points = a[np.where(encoding == i)]
        plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(
        codebook.prototypes[:, 0],
        codebook.prototypes[:, 1],
        marker='+'
    )
    plt.savefig("prototypes.png")


def _test_maddness(maddness_pkl_path=None, train_npy_path=None, test_npy_path=None):
    logger = config_logger("maddness", None)

    if train_npy_path is not None:
        dic = np.load(train_npy_path, allow_pickle=True)
        a, b = dic.item()["a"], dic.item()["b"]
    else:
        a = np.random.rand(512, 768)
        b = np.random.rand(768, 768)

    if maddness_pkl_path is not None:
        assert train_npy_path is not None
        with open(maddness_pkl_path, "rb") as f:
            maddness = pickle.load(f)
    else:
        maddness = Maddness(4, False)
        maddness.fit(a, b)

    if test_npy_path is not None:
        a = np.load(test_npy_path)
    else:
        a = np.random.rand(*a.shape)

    tp0 = time.time()
    out = maddness(a)
    tp1 = time.time()
    ans = np.dot(a, b)
    tp2 = time.time()

    error_rate = np.abs(out - ans) / (np.abs(ans) + 1e-3)
    error_rate = error_rate.flatten()
    plt.boxplot(error_rate)
    plt.savefig("error_rate.png")

    logger = logging.getLogger("maddness")
    logger.info(f"mean error rate: {np.mean(error_rate)}")
    logger.info(f"median error rate: {np.median(error_rate)}")
    logger.info(f"maddness time: {(tp1 - tp0) * 1e3}ms")
    logger.info(f"numpy time: {(tp2 - tp1) * 1e3}ms")


if __name__ == "__main__":
    _test_maddness()
