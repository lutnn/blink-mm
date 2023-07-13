import torch.nn.functional as F
import torch.utils.data
from torchaudio.datasets import SPEECHCOMMANDS
import torchvision.transforms as transforms
import torchaudio.transforms
from torch.utils.data import DistributedSampler


class FixAudioLength:
    def __call__(self, tensor):
        length = 16000
        if tensor.shape[1] > length:
            return tensor[:, :length]
        elif tensor.shape[1] < length:
            return F.pad(tensor, (0, length - tensor.shape[1]), mode="constant", value=0)
        else:
            return tensor


def get_transform():
    return transforms.Compose([
        FixAudioLength(),
        torchaudio.transforms.MelSpectrogram(16000, n_mels=32, n_fft=1024),
    ])


class SpeechCommands(torch.utils.data.Dataset):
    LABELS = [
        'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
        'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
    ]

    def __init__(self, root, subset, transform=None) -> None:
        super().__init__()

        self.transform = transform
        self.dataset = SPEECHCOMMANDS(root, download=False, subset=subset)

    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
        assert sample_rate == 16000
        label = self.LABELS.index(label)
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, label

    def __len__(self):
        return len(self.dataset)


def get_dataset_num_classes():
    return len(SpeechCommands.LABELS)


def get_dist_train_data_loader(rank, world_size, batch_size, root):
    train_ds = SpeechCommands(root, "training", get_transform())
    return torch.utils.data.DataLoader(
        train_ds, batch_size,
        sampler=DistributedSampler(train_ds, world_size, rank, shuffle=True),
        num_workers=16
    )


def get_dist_test_data_loader(rank, world_size, batch_size, root):
    test_ds = SpeechCommands(root, "testing", get_transform())
    return torch.utils.data.DataLoader(
        test_ds, batch_size,
        sampler=DistributedSampler(test_ds, world_size, rank, shuffle=False),
        num_workers=16
    )
