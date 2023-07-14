import blink_mm.data.cifar as cifar
from torchaudio.datasets import SPEECHCOMMANDS


if __name__ == "__main__":
    for dataset in ["cifar10", "gtsrb", "svhn"]:
        cifar.get_test_data_loader(1, dataset, "./datasets")
        cifar.get_train_data_loader(1, dataset, "./datasets")

    SPEECHCOMMANDS("./datasets", download=True, subset="testing")
    SPEECHCOMMANDS("./datasets", download=True, subset="training")
