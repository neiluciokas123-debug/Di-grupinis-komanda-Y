"""PATOBULINTAS EKSPERIMENTAS (Proposed): DistilBERT perkeliamasis mokymas (transfer learning).

Kodėl DistilBERT, o ne BERT/RoBERTa?
  * Projekte keliamas realaus laiko reikalavimas (< 200 ms vienam komentarui).
    DistilBERT yra ~40 % mažesnis ir ~60 % greitesnis už BERT-base, išlaikant ~97 % jo
    tikslumo (Sanh ir kt., 2019).
  * Duomenų > 5000, todėl pagal metodikos taisyklę tinka transformeris.

Modelis spėja 7 etiketes vienu metu (multi-label, sigmoid + BCE):
  aggressive + 6 Jigsaw kategorijos.

Paleidimas (rekomenduojama su GPU, pvz. Google Colab):
    python -m src.train_distilbert                      # 50 000 komentarų, 2 epochos
    python -m src.train_distilbert --max-train 0        # visas train rinkinys
    python -m src.train_distilbert --max-train 5000 --epochs 1   # greitas bandymas CPU
Rezultatas:  models/distilbert/ (modelis, tokenizer, thresholds.json, history.json)
             reports/figures/distilbert_training.png
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from src.data import load_split
from src.metrics import best_threshold, binary_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-train", type=int, default=50_000,
                   help="Kiek train komentarų naudoti (0 = visus). Imama stratifikuotai.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=C.MAX_LEN)
    p.add_argument("--warmup", type=float, default=0.06, help="Warmup žingsnių dalis")
    p.add_argument("--pos-weight", action="store_true",
                   help="Didesnis svoris retoms teigiamoms klasėms (disbalanso kompensavimas)")
    p.add_argument("--base-model", default=C.DISTILBERT_BASE)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, tokenizer, max_len, batch_size, shuffle):
    texts = df["comment_text"].tolist()
    labels = df[C.ALL_LABELS].values.astype(np.float32)

    def collate(idx):
        enc = tokenizer([texts[i] for i in idx], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels[idx])
        return enc

    return DataLoader(list(range(len(texts))), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate)


@torch.no_grad()
def predict_probs(model, loader, device, use_amp):
    model.eval()
    probs, losses = [], []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(**batch).logits
        losses.append(loss_fn(logits.float(), labels).item())
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.vstack(probs), float(np.mean(losses))


def plot_history(hist: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = hist["step"]
    axes[0].plot(steps, hist["train_loss"], alpha=0.35, label="train loss (žingsnis)")
    if len(hist["train_loss"]) >= 10:
        k = max(len(hist["train_loss"]) // 20, 2)
        smooth = np.convolve(hist["train_loss"], np.ones(k) / k, mode="valid")
        axes[0].plot(steps[k - 1:], smooth, label="train loss (slenkantis vidurkis)")
    axes[0].plot(hist["epoch_step"], hist["val_loss"], "o-", label="val loss")
    axes[0].set_xlabel("Žingsnis")
    axes[0].set_title("Nuostolių funkcija")
    axes[0].legend()

    ep = list(range(1, len(hist["val_f1"]) + 1))
    axes[1].plot(ep, hist["val_f1"], "o-", label="val F1 (agresyvus)")
    axes[1].plot(ep, hist["val_auc"], "s-", label="val ROC-AUC (agresyvus)")
    axes[1].set_xlabel("Epocha")
    axes[1].set_xticks(ep)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title("Validacijos metrikos")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "distilbert_training.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Įrenginys: {device}" + ("" if use_amp else "  (be GPU treniravimas bus lėtas – rekomenduojama Colab)"))

    train, val = load_split("train"), load_split("val")
    if args.max_train and args.max_train < len(train):
        train, _ = train_test_split(train, train_size=args.max_train,
                                    stratify=train[C.MAIN_LABEL], random_state=C.SEED)
    print(f"Train: {len(train)} (agresyvių {train[C.MAIN_LABEL].mean():.1%}), Val: {len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(C.ALL_LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(C.ALL_LABELS)),
        label2id={l: i for i, l in enumerate(C.ALL_LABELS)},
    ).to(device)

    train_dl = make_loader(train, tokenizer, args.max_len, args.batch_size, shuffle=True)
    val_dl = make_loader(val, tokenizer, args.max_len, args.batch_size * 2, shuffle=False)

    pos_weight = None
    if args.pos_weight:
        pos = train[C.ALL_LABELS].values.mean(axis=0).clip(1e-3, None)
        # šaknis iš neg/pos santykio – švelnesnis nei pilnas, kad modelis neperkompensuotų
        pos_weight = torch.tensor(np.sqrt((1 - pos) / pos), dtype=torch.float32, device=device)
        print("pos_weight:", np.round(pos_weight.cpu().numpy(), 2))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_steps * args.warmup), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = {"step": [], "train_loss": [], "epoch_step": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_f1, step, t0 = -1.0, 0, time.time()
    main_idx = C.ALL_LABELS.index(C.MAIN_LABEL)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            hist["step"].append(step)
            hist["train_loss"].append(loss.item())
            if step % 100 == 0:
                print(f"  epocha {epoch} žingsnis {step}/{total_steps}  loss={loss.item():.4f}  "
                      f"({time.time() - t0:.0f} s)")

        probs, val_loss = predict_probs(model, val_dl, device, use_amp)
        y_val = val[C.MAIN_LABEL].values
        thr = best_threshold(y_val, probs[:, main_idx])
        m = binary_metrics(y_val, probs[:, main_idx], thr)
        hist["epoch_step"].append(step)
        hist["val_loss"].append(val_loss)
        hist["val_f1"].append(m["F1"])
        hist["val_auc"].append(m["ROC-AUC"])
        print(f"Epocha {epoch}: val loss={val_loss:.4f}  F1={m['F1']:.3f}  ROC-AUC={m['ROC-AUC']:.3f}")

        if m["F1"] > best_f1:   # išsaugome geriausią epochą (early stopping principas)
            best_f1 = m["F1"]
            model.save_pretrained(C.DISTILBERT_DIR)
            tokenizer.save_pretrained(C.DISTILBERT_DIR)
            thresholds = {lab: best_threshold(val[lab].values, probs[:, i])
                          for i, lab in enumerate(C.ALL_LABELS)}
            with open(C.DISTILBERT_DIR / "thresholds.json", "w") as f:
                json.dump(thresholds, f, indent=1)
            print(f"  -> išsaugotas geriausias modelis (epocha {epoch})")

    meta = {**vars(args), "train_size": len(train), "best_val_f1": best_f1,
            "train_minutes": round((time.time() - t0) / 60, 1), "device": str(device)}
    with open(C.DISTILBERT_DIR / "history.json", "w") as f:
        json.dump({"meta": meta, **hist}, f)
    plot_history(hist)
    print(f"Baigta per {meta['train_minutes']} min. Geriausias val F1={best_f1:.3f}. Modelis: {C.DISTILBERT_DIR}")


if __name__ == "__main__":
    main()
