"""EKSPERIMENTŲ PALYGINIMAS (ataskaitos VII skyrius).

Visi modeliai vertinami TOJE PAČIOJE test aibėje, kurios nematė treniruojant
ir renkant slenksčius (slenksčiai parinkti val aibėje).

Paleidimas:  python -m src.evaluate
Rezultatas:  reports/metrics/comparison.{csv,md}       – pagrindinė lyginamoji lentelė
             reports/metrics/categories.csv            – F1 / ROC-AUC kiekvienai kategorijai
             reports/metrics/latency.csv               – inferencijos laikas (ms / komentarui)
             reports/metrics/examples_*.csv            – sėkmės ir nesėkmės pavyzdžiai
             reports/metrics/bias_test.csv             – tapatybės žodžių šališkumo testas
             reports/figures/confusion_matrices.png, roc_pr_curves.png,
             categories_f1.png, bias_test.png, metrics_comparison.png
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from src.data import load_split
from src.metrics import binary_metrics
from src.predict import (BaselinePredictor, DistilBertPredictor, MajorityPredictor,
                         RulePredictor)

sns.set_theme(style="whitegrid")
MAIN = 0  # C.ALL_LABELS[0] == "aggressive"

# Neutralūs sakiniai su tapatybės žodžiais. Idealiu atveju visų tikimybė turi būti maža.
IDENTITIES = ["gay", "straight", "lesbian", "transgender", "muslim", "christian", "jewish",
              "atheist", "black", "white", "asian", "african", "woman", "man", "immigrant",
              "lithuanian", "american", "disabled", "old", "young"]
TEMPLATES = ["I am a {} person.", "My best friend is {}.", "Being {} is perfectly normal.",
             "We had a nice dinner with our {} neighbours."]


def load_models() -> list:
    models = [MajorityPredictor()]
    for cls in (RulePredictor, BaselinePredictor, DistilBertPredictor):
        try:
            models.append(cls())
            print(f"Įkeltas modelis: {cls.name}")
        except Exception as e:  # modelis dar neištreniruotas arba nėra torch
            print(f"Praleidžiama {cls.name}: {type(e).__name__}: {str(e)[:120]}")
    return models


def comparison_table(models, probs, y) -> pd.DataFrame:
    rows = []
    for m in models:
        r = binary_metrics(y, probs[m.name][:, MAIN], m.thresholds[C.MAIN_LABEL])
        r["Modelis"] = m.name
        rows.append(r)
    df = pd.DataFrame(rows).set_index("Modelis")
    df.round(4).to_csv(C.METRICS_DIR / "comparison.csv")
    (C.METRICS_DIR / "comparison.md").write_text(df.round(3).to_markdown(), encoding="utf-8")

    plot_df = df[["Precision", "Recall", "F1", "PR-AUC"]].reset_index().melt(
        id_vars="Modelis", var_name="Metrika", value_name="Reikšmė")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=plot_df, x="Metrika", y="Reikšmė", hue="Modelis", ax=ax)
    ax.axhline(0.70, ls="--", c="k", lw=1)
    ax.text(3.45, 0.71, "F1 = 0.70 reikalavimas", ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Bazinis vs. patobulintas modelis (test aibė, klasė „agresyvus“)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "metrics_comparison.png", dpi=150)
    plt.close(fig)
    return df


def confusion_matrices(models, probs, y) -> None:
    ms = [m for m in models if not isinstance(m, MajorityPredictor)]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 4.2), squeeze=False)
    for ax, m in zip(axes[0], ms):
        pred = (probs[m.name][:, MAIN] >= m.thresholds[C.MAIN_LABEL]).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Neagres.", "Agres."], yticklabels=["Neagres.", "Agres."])
        ax.set_title(m.name)
        ax.set_xlabel("Prognozė")
        ax.set_ylabel("Tikroji klasė")
    fig.suptitle("Konfuzijos matricos (test aibė)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def roc_pr_curves(models, probs, y) -> None:
    ms = [m for m in models if m.has_categories]
    if not ms:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in ms:
        s = probs[m.name][:, MAIN]
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        a1.plot(fpr, tpr, label=m.name)
        a2.plot(r, p, label=m.name)
    a1.plot([0, 1], [0, 1], "k--", lw=1, label="Atsitiktinis")
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC kreivė")
    a2.axhline(y.mean(), c="k", ls="--", lw=1, label="Atsitiktinis")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall kreivė")
    a1.legend()
    a2.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "roc_pr_curves.png", dpi=150)
    plt.close(fig)


def category_table(models, probs, test) -> None:
    rows = []
    for m in [m for m in models if m.has_categories]:
        for i, lab in enumerate(C.ALL_LABELS):
            if test[lab].sum() == 0:
                continue
            r = binary_metrics(test[lab].values, probs[m.name][:, i], m.thresholds[lab])
            rows.append({"Modelis": m.name, "Kategorija": C.LABELS_LT[lab],
                         "Teigiamų test aibėje": int(test[lab].sum()),
                         "F1": r["F1"], "ROC-AUC": r["ROC-AUC"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.round(4).to_csv(C.METRICS_DIR / "categories.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=df, x="Kategorija", y="F1", hue="Modelis", ax=ax)
    ax.set_title("F1 pagal kategoriją (retos kategorijos – sunkiausios)")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "categories_f1.png", dpi=150)
    plt.close(fig)


def latency(models, test, n=200) -> None:
    """Vieno komentaro (batch=1) apdorojimo laikas CPU – realaus laiko reikalavimas < 200 ms."""
    sample = test["comment_text"].sample(min(n, len(test)), random_state=C.SEED).tolist()
    rows = []
    for m in [m for m in models if m.has_categories]:
        pred = m
        if isinstance(m, DistilBertPredictor) and m.device.type != "cpu":
            pred = DistilBertPredictor(device="cpu")
        pred.predict_proba(sample[:3])  # „apšildymas“
        times = []
        for t in sample:
            t0 = time.perf_counter()
            pred.predict_proba([t])
            times.append((time.perf_counter() - t0) * 1000)
        rows.append({"Modelis": m.name, "Mediana, ms": np.median(times),
                     "95 procentilis, ms": np.percentile(times, 95),
                     "Atitinka < 200 ms": bool(np.percentile(times, 95) < 200)})
    if rows:
        pd.DataFrame(rows).round(2).to_csv(C.METRICS_DIR / "latency.csv", index=False)


def examples(models, probs, test, k=15) -> None:
    """Sėkmės ir nesėkmės pavyzdžiai geriausiam (paskutiniam įkeltam) modeliui."""
    best = [m for m in models if m.has_categories]
    if not best:
        return
    m = best[-1]
    s = probs[m.name][:, MAIN]
    thr = m.thresholds[C.MAIN_LABEL]
    df = pd.DataFrame({"comment_text": test["comment_text"].str.slice(0, 300),
                       "tikroji": test[C.MAIN_LABEL].values, "tikimybe": s.round(3)})
    pred = (s >= thr).astype(int)
    fp = df[(pred == 1) & (df.tikroji == 0)].sort_values("tikimybe", ascending=False).head(k)
    fn = df[(pred == 0) & (df.tikroji == 1)].sort_values("tikimybe").head(k)
    tp = df[(pred == 1) & (df.tikroji == 1)].sort_values("tikimybe", ascending=False).head(k)
    fp.to_csv(C.METRICS_DIR / "examples_false_positives.csv", index=False)
    fn.to_csv(C.METRICS_DIR / "examples_false_negatives.csv", index=False)
    tp.to_csv(C.METRICS_DIR / "examples_true_positives.csv", index=False)

    # Kur patobulintas modelis teisus, o bazinis – klysta (ir atvirkščiai)
    if len(best) >= 2:
        b = best[0]
        pb = (probs[b.name][:, MAIN] >= b.thresholds[C.MAIN_LABEL]).astype(int)
        y = test[C.MAIN_LABEL].values
        d = df.assign(**{f"{b.name}": probs[b.name][:, MAIN].round(3)})
        d[(pred == y) & (pb != y)].head(k * 2).to_csv(C.METRICS_DIR / "examples_proposed_wins.csv", index=False)
        d[(pred != y) & (pb == y)].head(k * 2).to_csv(C.METRICS_DIR / "examples_baseline_wins.csv", index=False)
    print(f"Pavyzdžiai išsaugoti ({m.name}): FP={len(fp)}, FN={len(fn)}")


def bias_test(models) -> None:
    """Ar modelis neteisingai laiko agresyviais neutralius sakinius su tapatybės žodžiais?

    Žinoma Jigsaw duomenų problema: žodžiai „gay“, „muslim“, „black“ dažnai pasitaiko
    toksiškuose komentaruose, todėl modelis gali išmokti, kad pats žodis yra „blogas“
    (Dixon ir kt., 2018).
    """
    ms = [m for m in models if m.has_categories]
    if not ms:
        return
    rows = []
    for m in ms:
        for ident in IDENTITIES:
            sents = [t.format(ident) for t in TEMPLATES]
            p = m.predict_proba(sents)[:, MAIN]
            rows.append({"Modelis": m.name, "Tapatybė": ident, "Vid. tikimybė": p.mean(),
                         "Klaidingai pažymėta (iš 4)": int((p >= m.thresholds[C.MAIN_LABEL]).sum())})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(C.METRICS_DIR / "bias_test.csv", index=False)
    pivot = df.pivot(index="Tapatybė", columns="Modelis", values="Vid. tikimybė")
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7, 7))
    pivot.plot.barh(ax=ax)
    ax.axvline(C.ZONE_LOW, ls="--", c="orange", lw=1, label="Moderatoriaus zona")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Vidutinė „agresyvumo“ tikimybė NEUTRALIEMS sakiniams")
    ax.set_title("Šališkumo testas: tapatybės žodžiai")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "bias_test.png", dpi=150)
    plt.close(fig)


def main() -> None:
    test = load_split("test")
    y = test[C.MAIN_LABEL].values
    models = load_models()
    print(f"Test aibė: {len(test)} komentarų (agresyvių {y.mean():.1%})")

    probs = {}
    for m in models:
        t0 = time.time()
        probs[m.name] = m.predict_proba(test["comment_text"].tolist())
        print(f"  {m.name}: prognozės per {time.time() - t0:.1f} s")

    table = comparison_table(models, probs, y)
    confusion_matrices(models, probs, y)
    roc_pr_curves(models, probs, y)
    category_table(models, probs, test)
    latency(models, test)
    examples(models, probs, test)
    bias_test(models)

    print("\n=== Lyginamoji lentelė (test aibė, klasė „agresyvus“) ===")
    print(table.round(3).to_string())
    lat = C.METRICS_DIR / "latency.csv"
    if lat.exists():
        print("\n=== Inferencijos laikas (CPU, 1 komentaras) ===")
        print(pd.read_csv(lat).to_string(index=False))
    print(f"\nVisi rezultatai: {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
