from torch.utils.data import DistributedSampler, DataLoader

from transformers import AutoTokenizer
from datasets import load_dataset


CONFIG = {
    "columns": {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
    },
    "splits": {
        "cola": ["train", "validation"],
        "sst2": ["train", "validation"],
        "mrpc": ["train", "validation"],
        "qqp": ["train", "validation"],
        "stsb": ["train", "validation"],
        "mnli": ["train", "validation_matched", "validation_mismatched"],
        "qnli": ["train", "validation"],
        "rte": ["train", "validation"],
        "wnli": ["train", "validation"],
    },
    "num_labels": {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
    },
    "metrics": {
        "cola": "accuracy",
        "sst2": "accuracy",
        "mrpc": "accuracy",
        "qqp": "accuracy",
        "stsb": "pearson_corr",
        "mnli": "accuracy",
        "qnli": "accuracy",
        "rte": "accuracy",
        "wnli": "accuracy",
    }
}

SUBMIT_CONFIG = {
    ("cola", "validation"): ("cola", "test", "CoLA"),
    ("sst2", "validation"): ("sst-2", "test", "SST-2"),
    ("mrpc", "validation"): ("mrpc", "test", "MRPC"),
    ("qqp", "validation"): ("qqp", "test", "QQP"),
    ("stsb", "validation"): ("sts-b", "test", "STS-B"),
    ("mnli", "validation_matched"): ("mnli", "test_matched", "MNLI-m"),
    ("mnli", "validation_mismatched"): ("mnli-mm", "test_mismatched", "MNLI-mm"),
    ("qnli", "validation"): ("qnli", "test", "QNLI"),
    ("rte", "validation"): ("rte", "test", "RTE"),
    ("wnli", "validation"): ("wnli", "test", "WNLI"),
}


def build_dataset(tokenizer, name, split, cols):
    def tokenize(samples):
        return tokenizer(*[samples[col] for col in cols], padding="max_length", truncation=True, max_length=len(cols) * 64)

    ds = load_dataset("glue", name)[split]
    ds = ds.map(tokenize, batched=True)
    ds.set_format(
        type='torch',
        columns=["input_ids", "token_type_ids", "attention_mask", "label"]
    )
    return ds


def get_tokenizer(name="bert-base-uncased"):
    return AutoTokenizer.from_pretrained(name)


def get_dist_train_data_loader(rank, world_size, tokenizer, name, batch_size):
    return get_dist_data_loader(
        rank, world_size, tokenizer, name, CONFIG["splits"][name][0], batch_size, True)


def get_dist_test_data_loaders(rank, world_size, tokenizer, name, batch_size):
    data_loaders = {}
    for i in range(1, len(CONFIG["splits"][name])):
        split = CONFIG["splits"][name][i]
        data_loader = get_dist_data_loader(
            rank, world_size, tokenizer, name, split, batch_size, False)
        data_loaders[split] = data_loader
    return data_loaders


def get_dist_data_loader(rank, world_size, tokenizer, name, split, batch_size, shuffle):
    dataset = build_dataset(
        tokenizer, name, split, CONFIG["columns"][name])
    return DataLoader(
        dataset, batch_size=batch_size,
        sampler=DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
    )
