from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def evaluate_regression(preds: np.ndarray, targets: np.ndarray):
    # preds: (batch_size, 1)
    # targets: (batch_size, )
    return {
        "mae": np.mean(np.abs(preds.flatten() - targets))
    }


class RegressionCNNWrapper(nn.Module):
    def __init__(self, cnn, device):
        super().__init__()
        self.device = device
        self.cnn = cnn
        self.cnn.to(self.device)

    @staticmethod
    def train_step(model, data_batch, optimizer):
        optimizer.zero_grad()
        preds = model.forward(data_batch[0].to(model.device))
        num_samples = preds.size(0)
        labels = data_batch[1].to(model.device).to(torch.float32)
        loss = F.mse_loss(preds.flatten(), labels)
        loss.backward()
        mae = torch.mean(torch.abs(preds.flatten() - labels))
        optimizer.step()
        return {
            "log_vars": {
                "loss": loss.cpu().detach().numpy(),
                "mae": mae.cpu().detach().numpy()
            },
            "num_samples": num_samples
        }

    def forward(self, x):
        return self.cnn.forward(x)

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
        return evaluate_regression(preds.detach().cpu().numpy(), targets.detach().cpu().numpy())
