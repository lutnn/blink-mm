import os
import os.path as osp
from glob import glob
import zipfile
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from transformers import glue_processors

from blink_mm.networks.seq_models.amm_bert import AMMBERT
from blink_mm.networks.seq_models.bert import BERT
from blink_mm.data.glue import CONFIG, SUBMIT_CONFIG, build_dataset, get_tokenizer


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, _, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


def _find_best_lr_ckpt_path(ckpt_path, dataset_type, split, metric_name):
    pairs = []
    for lr_str in os.listdir(osp.join(ckpt_path, dataset_type)):
        paths = glob(
            f"{ckpt_path}/{dataset_type}/{lr_str}/{split}_best_{metric_name}_epoch_*.pth")
        assert len(paths) == 1
        path = paths[0]
        state_dict = torch.load(path)
        pairs.append((path, state_dict["meta"][split][metric_name]))
    return max(pairs, key=lambda p: p[1])[0]


def amm_bert(num_labels, device):
    return AMMBERT(num_labels, device)


def bert_half(num_labels, device):
    return BERT(num_labels, device, num_hidden_layers=6)


def eval_test_set(model_type, device_id, ckpt_path, output_dir):
    device = f"cuda:{device_id}"

    os.makedirs(output_dir)

    LABELS = {
        "mnli": ["entailment", "neutral", "contradiction"],
        "mnli-mm": ["entailment", "neutral", "contradiction"],
    }

    for dataset_type in CONFIG["num_labels"].keys():
        metric_name = CONFIG["metrics"][dataset_type]
        for split in CONFIG["splits"][dataset_type][1:]:
            model = globals()[model_type](
                CONFIG["num_labels"][dataset_type], device)
            path = _find_best_lr_ckpt_path(
                ckpt_path, dataset_type, split, metric_name)
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict["state_dict"])
            processor_name, test_split, file_name = \
                SUBMIT_CONFIG[(dataset_type, split)]

            tokenizer = get_tokenizer()
            dataset = build_dataset(
                tokenizer, dataset_type, test_split, CONFIG["columns"][dataset_type])
            test_data_loader = DataLoader(
                dataset, batch_size=32, shuffle=False)

            model.eval()
            preds = []
            with torch.no_grad():
                for data_batch in tqdm(test_data_loader, desc=f"evaluating {file_name}"):
                    x, _ = model.val_step(model, data_batch, None)
                    preds.append(x)
            preds = torch.cat([pred.logits for pred in preds], dim=0)
            preds = preds.cpu().detach().numpy()

            num_labels = preds.shape[-1]
            if num_labels >= 2:
                preds = np.argmax(preds, axis=-1)
            else:
                preds = np.squeeze(preds)

            if LABELS.get(processor_name, None) is not None:
                labels = LABELS[processor_name]
            else:
                processor = glue_processors[processor_name]()
                labels = processor.get_labels()

            if num_labels == 1:
                # FIXME
                preds[preds > 5] = 5
                preds[preds < 0] = 0
                preds = list(preds)
            else:
                preds = [labels[pred] for pred in preds]

            df = pd.DataFrame({
                'index': range(len(preds)),
                'prediction': preds
            })
            df.to_csv(f'{output_dir}/{file_name}.tsv', sep='\t', index=False)

    # predictions for ax
    tokenizer = get_tokenizer()
    dataset = build_dataset(
        tokenizer, "ax", "test", ["premise", "hypothesis"])
    preds = ["entailment" for _ in range(len(dataset))]
    df = pd.DataFrame({
        'index': range(len(preds)),
        'prediction': preds
    })
    df.to_csv(f'{output_dir}/AX.tsv', sep='\t', index=False)

    with zipfile.ZipFile(f"{output_dir}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(output_dir, zipf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--model-type", default="amm_bert")

    args = parser.parse_args()
    eval_test_set(args.model_type, args.device_id,
                  args.ckpt_path, args.output_dir)
