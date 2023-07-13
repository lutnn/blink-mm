import os
import os.path as osp
import random

import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DistributedSampler

from PIL import Image


def get_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5961, 0.4561, 0.3903],
                             std=[0.2592, 0.2312, 0.2268]),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5961, 0.4561, 0.3903],
                             std=[0.2592, 0.2312, 0.2268]),
    ])


class UTKFace(torch.utils.data.Dataset):
    def __init__(self, root, train, transform):
        super().__init__()

        self.transform = transform
        self.root = osp.join(root, "UTKFace")
        files = os.listdir(self.root)
        random.Random(10086).shuffle(files)
        split = int(len(files) * 0.8)
        if train:
            self.files = files[:split]
        else:
            self.files = files[split:]

    def __getitem__(self, index):
        filename = self.files[index]
        age = int(filename[:filename.find('_')])
        img = Image.open(osp.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, age

    def __len__(self):
        return len(self.files)


def get_dist_train_data_loader(rank, world_size, batch_size, root):
    train_ds = UTKFace(root=root, train=True, transform=get_train_transform())
    return torch.utils.data.DataLoader(
        train_ds, batch_size,
        sampler=DistributedSampler(train_ds, world_size, rank, shuffle=True),
        num_workers=16
    )


def get_dist_test_data_loader(rank, world_size, batch_size, root):
    test_ds = UTKFace(root=root, train=False, transform=get_test_transform())
    return torch.utils.data.DataLoader(
        test_ds, batch_size,
        sampler=DistributedSampler(test_ds, world_size, rank, shuffle=False),
        num_workers=16
    )
