from torch.utils.data import Dataset, DataLoader, DistributedSampler

from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, AutoTokenizer


class MLMDataset(Dataset):
    def __init__(self, hf_dataset_path, tokenizer):
        super().__init__()
        self.dataset = load_from_disk(hf_dataset_path)
        self.dataset.set_format(type='torch')
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_ids, labels = self.data_collator.torch_mask_tokens(
            sample["input_ids"],
            sample["special_tokens_mask"]
        )
        sample["input_ids"] = input_ids
        sample["labels"] = labels
        return sample


def get_tokenizer(name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(name)


def get_dist_train_data_loader(rank, world_size, tokenizer, batch_size, hf_dataset_path):
    dataset = MLMDataset(hf_dataset_path, tokenizer)
    return DataLoader(
        dataset, batch_size=batch_size,
        sampler=DistributedSampler(dataset, world_size, rank, shuffle=True),
        num_workers=4
    )
