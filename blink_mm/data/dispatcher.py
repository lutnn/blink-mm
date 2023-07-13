import qat.data.imagenet as imagenet

import blink_mm.data.cifar as cifar
import blink_mm.data.speech_commands as speech_commands
import blink_mm.data.utk_face as utk_face


def get_dist_data_loader(rank, world_size, batch_size, dataset, root):
    if dataset in ["cifar10", "cifar100", "svhn", "gtsrb"]:
        train_data_loader = cifar.get_dist_train_data_loader(
            rank, world_size, batch_size, dataset, root)
        test_data_loader = cifar.get_dist_test_data_loader(
            rank, world_size, batch_size, dataset, root)
    elif dataset in ["speech_commands"]:
        train_data_loader = speech_commands.get_dist_train_data_loader(
            rank, world_size, batch_size, root)
        test_data_loader = speech_commands.get_dist_test_data_loader(
            rank, world_size, batch_size, root)
    elif dataset in ["imagenet"]:
        train_data_loader = imagenet.get_dist_train_data_loader(
            rank, world_size, batch_size, root)
        test_data_loader = imagenet.get_dist_test_data_loader(
            rank, world_size, batch_size, root)
    elif dataset in ["utk_face"]:
        train_data_loader = utk_face.get_dist_train_data_loader(
            rank, world_size, batch_size, root)
        test_data_loader = utk_face.get_dist_test_data_loader(
            rank, world_size, batch_size, root)
    return train_data_loader, test_data_loader


def get_data_config(dataset):
    if dataset in ["cifar10", "cifar100", "svhn", "gtsrb"]:
        return {
            "num_classes": cifar.get_dataset_num_classes(dataset),
            "in_channels": 3,
        }
    elif dataset in ["speech_commands"]:
        return {
            "num_classes": speech_commands.get_dataset_num_classes(),
            "in_channels": 1,
        }
    elif dataset in ["imagenet"]:
        return {
            "num_classes": 1000,
            # We do not set in_channels because torchvision models do not accept this argument
        }
    elif dataset in ["utk_face"]:
        return {
            "num_classes": 1,
            # We do not set in_channels because torchvision models do not accept this argument
        }
