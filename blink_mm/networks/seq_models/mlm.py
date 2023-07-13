import torch.nn as nn

from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class MLM(nn.Module):
    def __init__(self, seq_model, device):
        super().__init__()
        self.seq_model = seq_model
        self.mlm_head = BertOnlyMLMHead(self.seq_model.transformer.config)

        self.to(device)
        self.device = device

    def forward(self, *args, **kwargs):
        sequence_output = self.seq_model.forward(
            *args, **kwargs,
            labels=None,
            output_hidden_states=True
        ).hidden_states[-1]
        return self.mlm_head(sequence_output)

    @staticmethod
    def train_step(model, data_batch, optimizer):
        optimizer.zero_grad()

        prediction_scores = model.forward(
            data_batch["input_ids"].to(model.device),
            data_batch["attention_mask"].to(model.device),
            data_batch["token_type_ids"].to(model.device),
        )
        num_samples = prediction_scores.size(0)

        loss_fct = nn.CrossEntropyLoss()
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, prediction_scores.shape[-1]),
            data_batch["labels"].to(model.device).view(-1)
        )

        masked_lm_loss.backward()
        optimizer.step()

        return {
            "log_vars": {
                "loss": masked_lm_loss.cpu().detach().numpy(),
            },
            "num_samples": num_samples
        }

    @staticmethod
    def val_step(model, data_batch, optimizer):
        prediction_scores = model.forward(
            data_batch["input_ids"].to(model.device),
            data_batch["attention_mask"].to(model.device),
            data_batch["token_type_ids"].to(model.device),
        )
        return (prediction_scores, data_batch["labels"])
