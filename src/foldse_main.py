"""
Export binarized ViT-SAE features to CSV and train FOLD-SE rule models on them.

Usage (from the repo root or src/):
    python src/foldse_main.py --fetch           # extract features -> foldse/
    python src/foldse_main.py --train           # send CSVs to FOLD-SE -> foldse_res/
    python src/foldse_main.py --fetch --train   # both

FOLD-SE credentials are read from the FOLDSE_USERNAME / FOLDSE_PASSWORD env vars.
"""
import os
import json
import time
import argparse
from types import SimpleNamespace

import torch
import requests
import pandas as pd

from utils.train import get_dataloader
from utils.sae_inference import load_vivit_sae_model


# Configuration
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model-vitsae")
FOLDSE_DATA_DIR = os.path.join(REPO_ROOT, "foldse")
FOLDSE_RESULT_DIR = os.path.join(REPO_ROOT, "foldse_res")

FOLDSE_URL = "http://ec2-52-0-60-249.compute-1.amazonaws.com/auth/foldmodel_multicategory/"
# Visit http://ec2-52-0-60-249.compute-1.amazonaws.com/example/ to view an example.

DATASETS = [
    "flowers102",
    "cars",
    "fgvc",
    "caltech101",
    "foods101",
    "suns397",
    "dtd",
    "eurosat",
    "pets",
]

FEATURE_THRESHOLD = 0.5  # SAE activation threshold for binarizing features
REQUEST_DELAY_SEC = 5    # pause between FOLD-SE requests


def _binarized_cls_features(sparse_logits: torch.Tensor) -> torch.Tensor:
    """Binarize the CLS-token SAE activations of a single-image batch."""
    feat = sparse_logits[:, 0].squeeze(0)
    return (feat >= FEATURE_THRESHOLD).to(torch.int).cpu()


def _save_as_str_csv(rows: list, cols: list, path: str):
    df = pd.DataFrame(rows, columns=cols).astype(str)
    df.to_csv(path, index=False)


def fetch_data(iteration: int, data_name: str):
    """
    Write binarized SAE features for one dataset to CSV.
    Train rows use the ground-truth label; test rows use the model's
    predicted label (so FOLD-SE explains the model, not the data).
    """
    train_dataloader, val_dataloader, _, num_classes, _ = get_dataloader(
        inference=False,
        iteration=iteration,
        train_batch=1,
        val_batch=1,
    )

    model_args = SimpleNamespace(
        vit_sae_model_pth=os.path.join(MODEL_DIR, data_name, "best_topk_model.pth")
    )
    model = load_vivit_sae_model(model_args, num_classes=num_classes)
    model.eval()
    device = next(model.parameters()).device

    os.makedirs(FOLDSE_DATA_DIR, exist_ok=True)

    with torch.no_grad():
        # Train split: features + ground-truth label
        rows = []
        for img, label in train_dataloader:
            _, sparse_logits, _, _ = model(img.to(device))
            feat = _binarized_cls_features(sparse_logits)
            rows.append(torch.cat([feat, label.cpu().to(torch.int)]).tolist())

        cols = [f"feature_{i + 1}" for i in range(feat.size(0))] + ["label"]
        _save_as_str_csv(rows, cols, os.path.join(FOLDSE_DATA_DIR, f"{data_name}_train.csv"))

        # Test split: features + model-predicted label
        rows = []
        for img, _ in val_dataloader:
            _, sparse_logits, _, cls_logits = model(img.to(device))
            feat = _binarized_cls_features(sparse_logits)
            predicted = cls_logits.argmax(dim=1)
            rows.append(torch.cat([feat, predicted.cpu().to(torch.int)]).tolist())

        _save_as_str_csv(rows, cols, os.path.join(FOLDSE_DATA_DIR, f"{data_name}_test.csv"))


def _load_str_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path).astype(str)


def build_foldse_payload(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Assemble the request body for the FOLD-SE multi-category endpoint."""
    return {
        "username": os.environ.get("FOLDSE_USERNAME", ""),  # registered email id
        "password": os.environ.get("FOLDSE_PASSWORD", ""),
        "data_frame_json": train_df.to_json(orient="split"),
        "test_data_frame_json": test_df.to_json(orient="split"),
        # Names of numerical / categorical features, comma separated
        "numattrs": "",
        "strattrs": ",".join(train_df.columns[:-1]),
        # Train/test split (ignored since a separate test set is provided)
        "hyp1": "0.8",
        # Level of exceptions ratio
        "hyp2": "1.0",
        # Tail ratio
        "hyp3": "0.0005",
        # Positive class label (binary classification only)
        "positive_value": "",
        "label_value": "label",
        # "local" returns the model as JSON in response['model_json']
        "save_model": "",
    }


def train_foldse(data_name: str):
    """
    Send one dataset to FOLD-SE and save the response.
    On success the response contains: rules, accuracy, f1_score, precision,
    recall, n_rules, n_preds, size, test_results (and model_json if saved).
    """
    train_df = _load_str_csv(os.path.join(FOLDSE_DATA_DIR, f"{data_name}_train.csv"))
    test_df = _load_str_csv(os.path.join(FOLDSE_DATA_DIR, f"{data_name}_test.csv"))

    payload = build_foldse_payload(train_df, test_df)
    response = requests.post(FOLDSE_URL, json=payload)

    try:
        response_obj = response.json()
    except ValueError:
        print(response)
        return

    if "error" not in response_obj:
        print("There was an error processing your request.")
        return
    if response_obj["error"] is not None:
        print("Error: ", response_obj["error"])
        return

    out_dir = os.path.join(FOLDSE_RESULT_DIR, data_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "full_res.json"), "w") as file:
        json.dump(response_obj, file)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="Extract SAE features to CSV")
    parser.add_argument("--train", action="store_true", help="Train FOLD-SE on extracted CSVs")
    args = parser.parse_args()

    if not (args.fetch or args.train):
        parser.error("specify at least one of --fetch / --train")

    for i, data_name in enumerate(DATASETS):
        print(data_name)
        if args.fetch:
            fetch_data(i, data_name)
        if args.train:
            train_foldse(data_name)
            time.sleep(REQUEST_DELAY_SEC)


if __name__ == "__main__":
    main()
