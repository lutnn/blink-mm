import argparse

from datasets import concatenate_datasets, load_dataset
from transformers import BertTokenizerFast

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path")

    args = parser.parse_args()

    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    wiki = wiki.remove_columns(
        [col for col in wiki.column_names if col != "text"])

    assert bookcorpus.features.type == wiki.features.type
    raw_dataset = concatenate_datasets([bookcorpus, wiki])

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def group_texts(examples):
        from itertools import chain

        max_length = 128

        examples = tokenizer(
            examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
        )
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length

        result = {
            k: [t[i: i + max_length]
                for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }

        return result

    tokenized_dataset = raw_dataset.map(
        group_texts, batched=True, remove_columns=raw_dataset.column_names)

    tokenized_dataset.save_to_disk(args.dataset_path)
