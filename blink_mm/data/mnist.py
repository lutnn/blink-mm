import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler


def get_train_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_dist_train_data_loader(rank, world_size, batch_size, root):
    train_ds = torchvision.datasets.MNIST(
        root=root, train=True,
        download=False, transform=get_train_transform()
    )
    return torch.utils.data.DataLoader(
        train_ds, batch_size,
        sampler=DistributedSampler(train_ds, world_size, rank, shuffle=True),
        num_workers=16
    )


def get_dist_test_data_loader(rank, world_size, batch_size, root):
    test_ds = torchvision.datasets.MNIST(
        root=root, train=False,
        download=False, transform=get_test_transform()
    )
    return torch.utils.data.DataLoader(
        test_ds, batch_size,
        sampler=DistributedSampler(test_ds, world_size, rank, shuffle=False),
        num_workers=16
    )
