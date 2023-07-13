from typing import List

import torch
import torch.nn as nn

from scipy import stats

from qat.networks.evaluation import evaluate_classification


class SeqClassificationNetwork(nn.Module):
    def __init__(self, seq_model, device):
        super().__init__()
        self.seq_model = seq_model

        self.to(device)
        self.device = device

    def forward(self, *args, **kwargs):
        return self.seq_model.forward(*args, **kwargs)

    @staticmethod
    def train_step(model, data_batch, optimizer):
        optimizer.zero_grad()

        outputs = model.forward(
            data_batch["input_ids"].to(model.device),
            data_batch["attention_mask"].to(model.device),
            data_batch["token_type_ids"].to(model.device),
            labels=data_batch["label"].to(model.device)
        )
        preds = outputs.logits
        num_samples = preds.size(0)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        num_labels = outputs.logits.shape[-1]
        if num_labels >= 2:
            correct = torch.sum(
                torch.argmax(preds, dim=1) == data_batch["label"].to(
                    model.device)
            ).cpu().detach().numpy()
            return {
                "log_vars": {
                    "loss": loss.cpu().detach().numpy(),
                    "accuracy": correct / num_samples,
                },
                "num_samples": num_samples
            }
        else:
            return {
                "log_vars": {
                    "loss": loss.cpu().detach().numpy(),
                },
                "num_samples": num_samples
            }

    @staticmethod
    def val_step(model, data_batch, optimizer):
        outputs = model.forward(
            data_batch["input_ids"].to(model.device),
            data_batch["attention_mask"].to(model.device),
            data_batch["token_type_ids"].to(model.device),
            labels=None
        )
        return (outputs, data_batch["label"])

    @staticmethod
    def get_input_tuple(model, data_batch):
        return (
            data_batch["input_ids"].to(model.device),
            data_batch["attention_mask"].to(model.device),
            data_batch["token_type_ids"].to(model.device),
        )

    @staticmethod
    def evaluate(preds: list, targets: List[torch.Tensor]) -> dict:
        preds = torch.cat([pred.logits for pred in preds], dim=0)
        targets = torch.cat(targets)
        num_labels = preds.shape[-1]
        if num_labels >= 2:
            return evaluate_classification(preds.cpu().numpy(), targets.cpu().numpy())
        else:
            preds = preds.reshape(-1).cpu().numpy()
            targets = targets.cpu().numpy()
            spearman_corr = stats.spearmanr(preds, targets).correlation
            pearson_corr = stats.pearsonr(preds, targets)[0]
            return {
                "spearman_corr": spearman_corr,
                "pearson_corr": pearson_corr
            }
