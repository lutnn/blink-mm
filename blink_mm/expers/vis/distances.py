import argparse
from itertools import islice

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from qat.networks.cnn_wrapper import CNNWrapper

from blink_mm.networks.lenet.amm_lenet import AMMLeNet5
from blink_mm.im2col import unfold
from blink_mm.data.mnist import get_dist_test_data_loader


def simulate_forward(amm_conv2d, x):
    cols = unfold(x, amm_conv2d.kernel_size,
                  amm_conv2d.stride, amm_conv2d.padding)
    x = cols.permute(0, 2, 1).flatten(0, 1)
    x = x.reshape(x.shape[0], amm_conv2d.ncodebooks, amm_conv2d.subvec_len)
    x = x.permute(1, 0, 2)
    dist = torch.cdist(x, amm_conv2d.centroids)
    argmin = dist.argmin(dim=-1)
    # (ncodebooks, bhw, subvec_len)
    # (ncodebooks, 16, subvec_len)
    # (ncodebooks, bhw)
    return x, amm_conv2d.centroids, argmin


def vis_lenet(model, test_data_loader, image_path):
    dst = []

    def forward_hook(module, input_tensors, output_tensors):
        dst.append(input_tensors[0])

    amm_conv2d = model.cnn.layer2[0]

    handle = amm_conv2d.register_forward_hook(forward_hook)

    model.eval()
    with torch.no_grad():
        for data_batch in islice(test_data_loader, 1):
            model.val_step(model, data_batch, None)

    handle.remove()

    input_tensor = torch.cat(dst, dim=0)
    with torch.no_grad():
        x, centroids, argmin = simulate_forward(amm_conv2d, input_tensor)

    x = x[0].detach().cpu().numpy()
    centroids = centroids[0].detach().cpu().numpy()
    argmin = argmin[0].detach().cpu().numpy()

    plt.scatter(x[:, 0], x[:, 1], c=argmin)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", color="red")

    plt.savefig(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--root", default="./datasets")

    args = parser.parse_args()

    test_data_loader = get_dist_test_data_loader(0, 1, 32, args.root)

    model = AMMLeNet5(vis_distances=True)
    model = CNNWrapper(model, "cuda:0")
    model.load_state_dict(torch.load(args.ckpt_path)["state_dict"])

    vis_lenet(model, test_data_loader, "lenet.png")
