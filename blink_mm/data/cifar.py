import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler


def get_cifar_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


def get_cifar_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


def get_gtsrb_train_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171),
                             (0.2672, 0.2564, 0.2629))
    ])


def get_gtsrb_test_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171),
                             (0.2672, 0.2564, 0.2629))
    ])


def get_train_transform(name):
    assert name in ["cifar10", "cifar100", "svhn", "gtsrb"]
    if name in ["cifar10", "cifar100", "svhn"]:
        return get_cifar_train_transform()
    elif name in ["gtsrb"]:
        return get_gtsrb_train_transform()


def get_test_transform(name):
    assert name in ["cifar10", "cifar100", "svhn", "gtsrb"]
    if name in ["cifar10", "cifar100", "svhn"]:
        return get_cifar_test_transform()
    elif name in ["gtsrb"]:
        return get_gtsrb_test_transform()


def _get_dataset(name):
    assert name in ["cifar10", "cifar100", "svhn", "gtsrb"]
    if name == "cifar10":
        return torchvision.datasets.CIFAR10
    elif name == "cifar100":
        return torchvision.datasets.CIFAR100
    elif name == "svhn":
        return torchvision.datasets.SVHN
    elif name == "gtsrb":
        return torchvision.datasets.GTSRB


def get_dataset_num_classes(name):
    assert name in ["cifar10", "cifar100", "svhn", "gtsrb"]
    if name == "cifar10":
        return 10
    elif name == "cifar100":
        return 100
    elif name == "svhn":
        return 10
    elif name == "gtsrb":
        return 43


def _get_dataset_split_kwargs(name, split):
    assert name in ["cifar10", "cifar100", "svhn", "gtsrb"]
    if name in ["cifar10", "cifar100"]:
        return {"train": split == "train"}
    elif name in ["svhn", "gtsrb"]:
        return {"split": split}


def get_train_data_loader(batch_size, dataset, root):
    ds = _get_dataset(dataset)(
        root=root, download=True, transform=get_train_transform(dataset),
        **_get_dataset_split_kwargs(dataset, "train"),
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        shuffle=True, num_workers=8
    )


def get_test_data_loader(batch_size, dataset, root):
    ds = _get_dataset(dataset)(
        root=root, download=True, transform=get_test_transform(dataset),
        **_get_dataset_split_kwargs(dataset, "test"),
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        shuffle=False, num_workers=8
    )


def get_dist_train_data_loader(rank, world_size, batch_size, dataset, root):
    train_ds = _get_dataset(dataset)(
        root=root, download=False, transform=get_train_transform(dataset),
        **_get_dataset_split_kwargs(dataset, "train"),
    )
    return torch.utils.data.DataLoader(
        train_ds, batch_size,
        sampler=DistributedSampler(train_ds, world_size, rank, shuffle=True),
        num_workers=16
    )


def get_dist_test_data_loader(rank, world_size, batch_size, dataset, root):
    test_ds = _get_dataset(dataset)(
        root=root, download=False, transform=get_test_transform(dataset),
        **_get_dataset_split_kwargs(dataset, "test"),
    )
    return torch.utils.data.DataLoader(
        test_ds, batch_size,
        sampler=DistributedSampler(test_ds, world_size, rank, shuffle=False),
        num_workers=16
    )
