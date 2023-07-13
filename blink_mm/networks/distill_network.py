from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from qat.networks.evaluation import evaluate_classification


class DistillNetwork(nn.Module):
    def __init__(self, student, teacher, device):
        super().__init__()
        self.student = student
        self.dic = {
            "teacher": teacher,
        }
        self.device = device
        self.student.to(self.device)
        self.dic["teacher"].to(self.device)
        self.dic["teacher"].eval()

    def forward(self, *args, **kwargs):
        return self.student(*args, **kwargs)

    @staticmethod
    def train_step(model, data_batch, optimizer):
        teacher = (model.module if dist.is_initialized()
                   else model).dic["teacher"]

        optimizer.zero_grad()
        preds = model.forward(data_batch[0].to(model.device))
        correct = torch.sum(
            torch.argmax(preds, dim=1) == data_batch[1].to(model.device)
        ).cpu().detach().numpy()
        num_samples = preds.size(0)
        hard_target = data_batch[1].to(model.device)
        with torch.no_grad():
            soft_target = F.softmax(teacher(
                data_batch[0].to(model.device)).to(model.device), dim=-1)
        loss = F.cross_entropy(preds, hard_target) + \
            F.cross_entropy(preds, soft_target)
        loss.backward()
        optimizer.step()
        return {
            "log_vars": {
                "loss": loss.cpu().detach().numpy(),
                "accuracy": correct / num_samples,
            },
            "num_samples": num_samples
        }

    @staticmethod
    def val_step(model, data_batch, optimizer):
        preds = model.forward(data_batch[0].to(model.device))
        return (preds, data_batch[1])

    @staticmethod
    def get_input_tuple(model, data_batch):
        return (data_batch[0].to(model.device),)

    @staticmethod
    def evaluate(preds: List[torch.Tensor], targets: List[torch.Tensor]) -> dict:
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)
        return evaluate_classification(preds.detach().cpu().numpy(), targets.detach().cpu().numpy())
