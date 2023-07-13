import argparse
from itertools import islice

import torch

import matplotlib.pyplot as plt

from qat.networks.cnn_wrapper import CNNWrapper

from blink_mm.networks.lenet.lenet import LeNet5
from blink_mm.networks.lenet.amm_lenet import AMMLeNet5
from blink_mm.data.mnist import get_dist_test_data_loader


def vis_lenet(model, test_data_loader, image_path):
    dst = []

    def forward_hook(module, input_tensors, output_tensors):
        dst.append(output_tensors)

    handle = model.cnn.fc1.register_forward_hook(forward_hook)

    ys = []

    model.eval()
    with torch.no_grad():
        for data_batch in islice(test_data_loader, 128):
            model.val_step(model, data_batch, None)
            ys.append(data_batch[1])

    handle.remove()

    output_tensor = torch.cat(dst, dim=0)
    ys = torch.cat(ys, dim=0)
    xs = output_tensor.detach().cpu().numpy()
    ys = ys.detach().cpu().numpy()

    plt.scatter(xs[:, 0], xs[:, 1], c=ys)
    plt.savefig(image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", default="")
    parser.add_argument("--amm-ckpt-path", default="")
    parser.add_argument("--root", default="./datasets")

    args = parser.parse_args()

    test_data_loader = get_dist_test_data_loader(0, 1, 32, args.root)

    model = LeNet5(vis_features=True)
    model = CNNWrapper(model, "cuda:0")
    model.load_state_dict(torch.load(args.ckpt_path)["state_dict"])

    vis_lenet(model, test_data_loader, "lenet.png")

    model = AMMLeNet5(vis_features=True)
    model = CNNWrapper(model, "cuda:0")
    model.load_state_dict(torch.load(args.amm_ckpt_path)["state_dict"])

    vis_lenet(model, test_data_loader, "amm_lenet.png")
