"""Žvalgomoji duomenų analizė (EDA) ataskaitos IV skyriui.

Paleidimas:  python -m src.eda
Rezultatas:  reports/figures/eda_*.png, reports/metrics/eda_summary.csv,
             reports/metrics/toxic_words.csv
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from src.data import load_split
from src.text_utils import tokenize

sns.set_theme(style="whitegrid")


def label_distribution(df: pd.DataFrame) -> None:
    counts = df[C.ALL_LABELS].sum().rename(index=C.LABELS_LT)
    share = (counts / len(df) * 100).round(2)
    pd.DataFrame({"kiekis": counts, "procentai": share}).to_csv(C.METRICS_DIR / "eda_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#c0392b")
    for i, (v, s) in enumerate(zip(counts.values, share.values)):
        ax.text(v, i, f"  {v} ({s}%)", va="center", fontsize=9)
    ax.set_title(f"Etikečių pasiskirstymas (train, n={len(df)})")
    ax.set_xlabel("Komentarų skaičius")
    ax.set_xlim(0, counts.max() * 1.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_labels.png", dpi=150)
    plt.close(fig)


def class_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    vals = df[C.MAIN_LABEL].value_counts().sort_index()
    ax.bar(["Neagresyvus", "Agresyvus"], vals.values, color=["#27ae60", "#c0392b"])
    for i, v in enumerate(vals.values):
        ax.text(i, v, f"{v}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Klasių disbalansas")
    ax.set_ylim(0, vals.max() * 1.2)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_balance.png", dpi=150)
    plt.close(fig)


def length_histogram(df: pd.DataFrame) -> None:
    n_words = df["comment_text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, color, name in ((0, "#27ae60", "Neagresyvūs"), (1, "#c0392b", "Agresyvūs")):
        ax.hist(n_words[df[C.MAIN_LABEL] == lab].clip(upper=400), bins=60, alpha=0.6,
                density=True, color=color, label=name)
    ax.axvline(C.MAX_LEN * 0.75, ls="--", c="k", lw=1)
    ax.text(C.MAX_LEN * 0.75, ax.get_ylim()[1] * 0.9, f"  ~{C.MAX_LEN} žetonų riba", fontsize=8)
    ax.set_title("Komentarų ilgis žodžiais (apkarpyta ties 400)")
    ax.set_xlabel("Žodžių skaičius")
    ax.set_ylabel("Tankis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_length.png", dpi=150)
    plt.close(fig)

    desc = n_words.groupby(df[C.MAIN_LABEL]).describe().round(1)
    desc.index = ["Neagresyvūs", "Agresyvūs"]
    desc.to_csv(C.METRICS_DIR / "eda_length_stats.csv")


def caps_and_punctuation(df: pd.DataFrame) -> None:
    """Ar agresyvūs komentarai dažniau RAŠOMI DIDŽIOSIOMIS ir su !!!"""
    txt = df["comment_text"]
    letters = txt.str.count(r"[A-Za-z]").replace(0, 1)
    feats = pd.DataFrame({
        "Didžiųjų raidžių dalis": txt.str.count(r"[A-Z]") / letters,
        "Šauktukų skaičius": txt.str.count("!"),
    })
    out = feats.groupby(df[C.MAIN_LABEL].map({0: "Neagresyvūs", 1: "Agresyvūs"})).mean().round(3)
    out.to_csv(C.METRICS_DIR / "eda_style.csv")


def relative_word_frequency(df: pd.DataFrame, min_count: int = 30, top_n: int = 30) -> pd.DataFrame:
    """Žodžiai, kurie agresyviuose komentaruose sutinkami daug dažniau nei neagresyviuose.

    Santykis = P(žodis | agresyvus) / P(žodis | neagresyvus), su +1 glodinimu.
    Šis sąrašas naudojamas ir taisyklėmis grįstam baziniam modeliui (src/train_baseline.py).
    """
    tox, clean = Counter(), Counter()
    for text, lab in zip(df["comment_text"], df[C.MAIN_LABEL]):
        words = set(tokenize(text))  # dokumento dažnis: ar žodis yra komentare
        (tox if lab else clean).update(words)

    n_tox = max(df[C.MAIN_LABEL].sum(), 1)
    n_clean = max((1 - df[C.MAIN_LABEL]).sum(), 1)
    rows = []
    for w in set(tox) | set(clean):
        t, c = tox[w], clean[w]
        if t + c < min_count:
            continue
        ratio = ((t + 1) / n_tox) / ((c + 1) / n_clean)
        rows.append((w, t, c, ratio))
    res = pd.DataFrame(rows, columns=["word", "in_aggressive", "in_clean", "ratio"])
    res = res.sort_values("ratio", ascending=False).reset_index(drop=True)
    res.to_csv(C.METRICS_DIR / "toxic_words.csv", index=False)

    top = res.head(top_n)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["word"][::-1], top["ratio"][::-1], color="#c0392b")
    ax.set_xscale("log")
    ax.set_title(f"Top {top_n} žodžių, būdingų agresyviems komentarams")
    ax.set_xlabel("Santykinis dažnis (log skalė)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_toxic_words.png", dpi=150)
    plt.close(fig)
    return res


def label_correlation(df: pd.DataFrame) -> None:
    corr = df[C.CATEGORY_LABELS].corr().rename(index=C.LABELS_LT, columns=C.LABELS_LT)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, ax=ax)
    ax.set_title("Kategorijų koreliacija")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_label_corr.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_split("train")
    label_distribution(df)
    class_balance(df)
    length_histogram(df)
    caps_and_punctuation(df)
    label_correlation(df)
    words = relative_word_frequency(df)
    print("EDA baigta. Grafikai:", C.FIGURES_DIR)
    print("Dažniausiai agresyviems būdingi žodžiai:", ", ".join(words["word"].head(15)))


if __name__ == "__main__":
    main()
  ""Žvalgomoji duomenų analizė (EDA) ataskaitos IV skyriui.

Paleidimas:  python -m src.eda
Rezultatas:  reports/figures/eda_*.png, reports/metrics/eda_summary.csv,
             reports/metrics/toxic_words.csv
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from src.data import load_split
from src.text_utils import tokenize

sns.set_theme(style="whitegrid")


def label_distribution(df: pd.DataFrame) -> None:
    counts = df[C.ALL_LABELS].sum().rename(index=C.LABELS_LT)
    share = (counts / len(df) * 100).round(2)
    pd.DataFrame({"kiekis": counts, "procentai": share}).to_csv(C.METRICS_DIR / "eda_summary.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#c0392b")
    for i, (v, s) in enumerate(zip(counts.values, share.values)):
        ax.text(v, i, f"  {v} ({s}%)", va="center", fontsize=9)
    ax.set_title(f"Etikečių pasiskirstymas (train, n={len(df)})")
    ax.set_xlabel("Komentarų skaičius")
    ax.set_ylabel("")
    ax.set_xlim(0, counts.max() * 1.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_labels.png", dpi=150)
    plt.close(fig)


def class_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    vals = df[C.MAIN_LABEL].value_counts().sort_index()
    ax.bar(["Neagresyvus", "Agresyvus"], vals.values, color=["#27ae60", "#c0392b"])
    for i, v in enumerate(vals.values):
        ax.text(i, v, f"{v}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Klasių disbalansas")
    ax.set_ylim(0, vals.max() * 1.2)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_balance.png", dpi=150)
    plt.close(fig)


def length_histogram(df: pd.DataFrame) -> None:
    n_words = df["comment_text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    for lab, color, name in ((0, "#27ae60", "Neagresyvūs"), (1, "#c0392b", "Agresyvūs")):
        ax.hist(n_words[df[C.MAIN_LABEL] == lab].clip(upper=400), bins=60, alpha=0.6,
                density=True, color=color, label=name)
    ax.axvline(C.MAX_LEN * 0.75, ls="--", c="k", lw=1)
    ax.text(C.MAX_LEN * 0.75, ax.get_ylim()[1] * 0.9, f"  ~{C.MAX_LEN} žetonų riba", fontsize=8)
    ax.set_title("Komentarų ilgis žodžiais (apkarpyta ties 400)")
    ax.set_xlabel("Žodžių skaičius")
    ax.set_ylabel("Tankis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_length.png", dpi=150)
    plt.close(fig)

    desc = n_words.groupby(df[C.MAIN_LABEL]).describe().round(1)
    desc.index = ["Neagresyvūs", "Agresyvūs"]
    desc.to_csv(C.METRICS_DIR / "eda_length_stats.csv")


def caps_and_punctuation(df: pd.DataFrame) -> None:
    """Ar agresyvūs komentarai dažniau RAŠOMI DIDŽIOSIOMIS ir su !!!"""
    txt = df["comment_text"]
    letters = txt.str.count(r"[A-Za-z]").replace(0, 1)
    feats = pd.DataFrame({
        "Didžiųjų raidžių dalis": txt.str.count(r"[A-Z]") / letters,
        "Šauktukų skaičius": txt.str.count("!"),
    })
    out = feats.groupby(df[C.MAIN_LABEL].map({0: "Neagresyvūs", 1: "Agresyvūs"})).mean().round(3)
    out.to_csv(C.METRICS_DIR / "eda_style.csv")


def relative_word_frequency(df: pd.DataFrame, min_count: int = 30, top_n: int = 30) -> pd.DataFrame:
    """Žodžiai, kurie agresyviuose komentaruose sutinkami daug dažniau nei neagresyviuose.

    Santykis = P(žodis | agresyvus) / P(žodis | neagresyvus), su +1 glodinimu.
    Šis sąrašas naudojamas ir taisyklėmis grįstam baziniam modeliui (src/train_baseline.py).
    """
    tox, clean = Counter(), Counter()
    for text, lab in zip(df["comment_text"], df[C.MAIN_LABEL]):
        words = set(tokenize(text))  # dokumento dažnis: ar žodis yra komentare
        (tox if lab else clean).update(words)

    n_tox = max(df[C.MAIN_LABEL].sum(), 1)
    n_clean = max((1 - df[C.MAIN_LABEL]).sum(), 1)
    rows = []
    for w in set(tox) | set(clean):
        t, c = tox[w], clean[w]
        if t + c < min_count:
            continue
        ratio = ((t + 1) / n_tox) / ((c + 1) / n_clean)
        rows.append((w, t, c, ratio))
    res = pd.DataFrame(rows, columns=["word", "in_aggressive", "in_clean", "ratio"])
    res = res.sort_values("ratio", ascending=False).reset_index(drop=True)
    res.to_csv(C.METRICS_DIR / "toxic_words.csv", index=False)

    top = res.head(top_n)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["word"][::-1], top["ratio"][::-1], color="#c0392b")
    ax.set_xscale("log")
    ax.set_title(f"Top {top_n} žodžių, būdingų agresyviems komentarams")
    ax.set_xlabel("Santykinis dažnis (log skalė)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_toxic_words.png", dpi=150)
    plt.close(fig)
    return res


def label_correlation(df: pd.DataFrame) -> None:
    corr = df[C.CATEGORY_LABELS].corr().rename(index=C.LABELS_LT, columns=C.LABELS_LT)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, ax=ax)
    ax.set_title("Kategorijų koreliacija")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "eda_label_corr.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_split("train")
    label_distribution(df)
    class_balance(df)
    length_histogram(df)
    caps_and_punctuation(df)
    label_correlation(df)
    words = relative_word_frequency(df)
    print("EDA baigta. Grafikai:", C.FIGURES_DIR)
    print("Dažniausiai agresyviems būdingi žodžiai:", ", ".join(words["word"].head(15)))
    """PAAIŠKINAMUMAS (XAI): kodėl modelis komentarą laiko agresyviu?

Trys metodai:
  1) Okliuzija (leave-one-word-out) – pašaliname po vieną žodį ir žiūrime, kiek nukrenta
     agresyvumo tikimybė. Veikia su bet kuriuo modeliu, greitas – naudojamas sąsajoje.
  2) LIME (Ribeiro ir kt., 2016) – lokalus tiesinis aproksimavimas aplink komentarą.
  3) Globalus Baseline paaiškinimas – didžiausi LogisticRegression koeficientai.

Paleidimas:  python -m src.explain                       # LIME + globalūs grafikai
             python -m src.explain "you are so stupid"    # vieno komentaro paaiškinimas
Rezultatas:  reports/xai/lime_*.html, reports/figures/xai_*.png
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

_WORD = re.compile(r"\S+")


def occlusion(predictor, text: str, max_words: int = 120) -> list[tuple[str, float]]:
    """Kiekvienam žodžiui: logit(visas tekstas) - logit(tekstas be šio žodžio).

    Teigiama reikšmė => žodis didina agresyvumą, neigiama => mažina.
    Skaičiuojama log-odds skalėje, nes tikimybėms esant arti 0 ar 1 („prisotinimas“)
    jų skirtumas būtų ~0, net jei žodis svarbus (pvz., du įžeidimai viename sakinyje).
    """
    words = _WORD.findall(text)[:max_words]
    if not words:
        return []
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    if hasattr(predictor, "predict_logit"):
        logit = np.asarray(predictor.predict_logit(variants), dtype=float)
    else:
        p = np.clip(predictor.predict_proba(variants)[:, 0], 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
    return [(w, float(logit[0] - logit[i + 1])) for i, w in enumerate(words)]


def lime_explanation(predictor, text: str, num_features: int = 10, num_samples: int = 500):
    from lime.lime_text import LimeTextExplainer

    def fn(texts):
        p = predictor.predict_proba(list(texts))[:, 0]
        return np.column_stack([1 - p, p])

    explainer = LimeTextExplainer(class_names=["Neagresyvus", "Agresyvus"], random_state=C.SEED)
    return explainer.explain_instance(text, fn, num_features=num_features, num_samples=num_samples)


def plot_word_weights(weights, title, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    weights = sorted(weights, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(weights) + 1.2))
    ax.barh([w for w, _ in weights], [v for _, v in weights],
            color=["#c0392b" if v > 0 else "#27ae60" for _, v in weights])
    ax.axvline(0, c="k", lw=0.8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def global_baseline_importance(top_n: int = 25) -> None:
    """Kurie žodžiai TF-IDF + LR modeliui labiausiai reiškia „agresyvus“ / „neagresyvus“."""
    import joblib
    b = joblib.load(C.BASELINE_DIR / "tfidf_lr.joblib")
    clf = b["models"][C.MAIN_LABEL]
    n_word = len(b["word_vec"].vocabulary_)
    coef = clf.coef_[0][:n_word]  # tik žodžių dalis (simbolių n-gramos sunkiai skaitomos)
    vocab = np.array(b["word_vec"].get_feature_names_out())
    order = np.argsort(coef)
    top = [(vocab[i], coef[i]) for i in order[-top_n:]] + [(vocab[i], coef[i]) for i in order[:10]]
    plot_word_weights(top, "TF-IDF + LogReg: didžiausi žodžių koeficientai\n(raudona – agresyvumas, žalia – neutralumas)",
                      C.FIGURES_DIR / "xai_baseline_global.png")


def main(texts=None) -> None:
    from src.predict import available_models
    models = available_models()
    if not models:
        sys.exit("Nėra ištreniruotų modelių.")
    name, cls = next(iter(models.items()))  # DistilBERT, jei yra; kitaip Baseline
    predictor = cls()
    out_dir = C.REPORTS_DIR / "xai"
    out_dir.mkdir(exist_ok=True)

    if "TF-IDF + LogReg" in models:
        global_baseline_importance()

    if texts is None:
        from src.data import load_split
        test = load_split("test")
        # keli agresyvūs ir keli neagresyvūs pavyzdžiai iš test aibės (trumpi – kad būtų įskaitoma)
        short = test[test["comment_text"].str.len().between(30, 250)]
        texts = (short[short[C.MAIN_LABEL] == 1].sample(3, random_state=C.SEED)["comment_text"].tolist()
                 + short[short[C.MAIN_LABEL] == 0].sample(2, random_state=C.SEED)["comment_text"].tolist())

    for i, t in enumerate(texts):
        p = predictor.predict_proba([t])[0, 0]
        occ = occlusion(predictor, t)
        top = sorted(occ, key=lambda x: abs(x[1]), reverse=True)[:12]
        print(f"\n[{name}] p(agresyvus)={p:.3f}  {t[:100]!r}")
        print("  Okliuzija:", ", ".join(f"{w}:{v:+.2f}" for w, v in top[:8]))
        plot_word_weights(top, f"{name}, okliuzija, p={p:.2f}\n{t[:80]}",
                          C.FIGURES_DIR / f"xai_occlusion_{i}.png")
        try:
            exp = lime_explanation(predictor, t)
            exp.save_to_file(str(out_dir / f"lime_{i}.html"))
            print("  LIME:     ", ", ".join(f"{w}:{v:+.2f}" for w, v in exp.as_list()[:8]))
        except ImportError:
            print("  (LIME neįdiegtas: pip install lime)")
    print(f"\nXAI rezultatai: {out_dir} ir {C.FIGURES_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)



if __name__ == "__main__":
    main()

