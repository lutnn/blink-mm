import torch.nn as nn

from transformers import AutoModelForSequenceClassification


class BERT(nn.Module):
    def __init__(
        self, num_labels, name="bert-base-uncased",
        num_hidden_layers=12, torchscript=False
    ):
        super().__init__()
        self.torchscript = torchscript
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=num_labels, num_hidden_layers=num_hidden_layers,
            torchscript=torchscript
        )

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        dic = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            **kwargs
        }
        if self.torchscript and "labels" in dic:
            dic.pop("labels")
        return self.transformer(**dic)
