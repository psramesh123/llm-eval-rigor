"""
System 2: fine-tuned transformer encoder.

Deliberately written with an explicit training loop and a hand-rolled
classification head rather than AutoModelForSequenceClassification + Trainer,
so the full pipeline is visible:

    Dataset -> DataLoader -> tokenization -> dynamic padding -> batching ->
    forward pass -> cross-entropy loss -> backward pass -> gradient clipping ->
    optimizer step -> LR schedule -> validation -> checkpoint selection -> inference

This is FINE-TUNING a pre-trained encoder, not training from scratch. The
encoder weights are updated, but they start from distilbert-base-uncased.

Training is a separate step (run on GPU) that writes a checkpoint. Inference
loads that checkpoint and implements the same System interface as every other
approach, so the evaluation layer never knows what produced a prediction.

Epoch selection uses VAL, carved from TRAIN. It never touches DEV or EVAL.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .base import Prediction, System

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64          # utterances are ~15 tokens; 64 is generous
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 5
WARMUP_FRAC = 0.1
SEED = 20260903
CKPT = Path("artifacts/transformer")


def get_device() -> str:
    """cuda on Colab/NVIDIA, mps on Apple Silicon, cpu otherwise."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Dataset: turns (text, label) into tensors. No padding here — the collate
# function pads each batch to its own longest sequence, which is faster than
# padding everything to MAX_LEN.
# ----------------------------------------------------------------------------
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id):
        self.texts = list(texts)
        self.labels = None if labels is None else [label2id[l] for l in labels]
        self.tok = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        enc = self.tok(self.texts[i], truncation=True, max_length=MAX_LEN)
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        if self.labels is not None:
            item["label"] = self.labels[i]
        return item


def make_collate(pad_id: int):
    """Dynamic padding: pad each batch to its own max length."""
    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        ids, mask, labels = [], [], []
        for b in batch:
            pad = longest - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * pad)
            mask.append(b["attention_mask"] + [0] * pad)
            if "label" in b:
                labels.append(b["label"])
        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }
        if labels:
            out["label"] = torch.tensor(labels, dtype=torch.long)
        return out
    return collate


# ----------------------------------------------------------------------------
# Model: pre-trained encoder + a classification head written here, so the
# architecture isn't hidden inside a library convenience class.
# ----------------------------------------------------------------------------
class IntentClassifier(nn.Module):
    def __init__(self, n_classes: int, model_name: str = MODEL_NAME, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean-pool over real tokens (ignore padding) rather than taking [CLS].
        # DistilBERT has no next-sentence objective, so [CLS] is not a trained
        # sentence summary; mean pooling is the stronger default here.
        hidden = out.last_hidden_state                       # (B, T, H)
        mask = attention_mask.unsqueeze(-1).float()          # (B, T, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(self.dropout(pooled))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def evaluate(model, loader, device) -> tuple[float, float, list[int]]:
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            preds.extend(logits.argmax(-1).cpu().tolist())
            if "label" in batch:
                golds.extend(batch["label"].tolist())
    if not golds:
        return 0.0, 0.0, preds
    acc = float(np.mean(np.array(preds) == np.array(golds)))
    macro_f1 = f1_score(golds, preds, average="macro")
    return acc, macro_f1, preds


def train(train_csv="data/processed/train_set.csv",
          val_csv="data/processed/val_set.csv",
          labels_json="data/processed/labels.json") -> None:
    set_seed()
    device = get_device()
    print(f"device: {device}")

    labels = json.loads(Path(labels_json).read_text())
    label2id = {l: i for i, l in enumerate(labels)}

    tr = pd.read_csv(train_csv)
    va = pd.read_csv(val_csv)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = make_collate(tok.pad_token_id)

    train_loader = DataLoader(
        IntentDataset(tr.text, tr.category, tok, label2id),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(
        IntentDataset(va.text, va.category, tok, label2id),
        batch_size=64, shuffle=False, collate_fn=collate)

    model = IntentClassifier(len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_FRAC),
        num_training_steps=total_steps)

    CKPT.mkdir(parents=True, exist_ok=True)
    best_f1, history = -1.0, []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device))
            loss = criterion(logits, batch["label"].to(device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += loss.item()

        val_acc, val_f1, _ = evaluate(model, val_loader, device)
        train_loss = running / len(train_loader)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_acc": val_acc, "val_macro_f1": val_f1})
        print(f"epoch {epoch}  loss {train_loss:.4f}  "
              f"val_acc {val_acc:.4f}  val_macro_f1 {val_f1:.4f}")

        # Checkpoint selection on VAL macro-F1. Never on DEV or EVAL.
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CKPT / "model.pt")
            tok.save_pretrained(CKPT)
            print(f"  saved (best val_macro_f1 {best_f1:.4f})")

    (CKPT / "training_log.json").write_text(json.dumps({
        "model_name": MODEL_NAME, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
        "lr": LR, "epochs": EPOCHS, "warmup_frac": WARMUP_FRAC, "seed": SEED,
        "best_val_macro_f1": best_f1, "history": history,
    }, indent=2))
    print(f"\nbest val macro-F1 {best_f1:.4f} -> {CKPT}/model.pt")


# ----------------------------------------------------------------------------
# Inference: same System interface as every other approach
# ----------------------------------------------------------------------------
class TransformerClassifier(System):
    name = "transformer"

    def __init__(self, ckpt: Path = CKPT, labels_json="data/processed/labels.json"):
        self.device = get_device()
        self.labels = json.loads(Path(labels_json).read_text())
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.model = IntentClassifier(len(self.labels))
        self.model.load_state_dict(
            torch.load(Path(ckpt) / "model.pt", map_location=self.device))
        self.model.to(self.device).eval()
        self.collate = make_collate(self.tok.pad_token_id)
        self.log = json.loads((Path(ckpt) / "training_log.json").read_text())

    def config(self) -> dict:
        return {k: v for k, v in self.log.items() if k != "history"}

    def fit(self, texts, labels) -> None:
        return None   # training is a separate offline step

    def predict(self, ids, texts) -> list[Prediction]:
        ds = IntentDataset(texts, None, self.tok, {})
        loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=self.collate)
        t0 = time.perf_counter()
        _, _, idxs = evaluate(self.model, loader, self.device)
        per_example_ms = (time.perf_counter() - t0) * 1000 / len(texts)
        return [
            Prediction(eid, self.labels[i], per_example_ms)
            for eid, i in zip(ids, idxs)
        ]


if __name__ == "__main__":
    train()
