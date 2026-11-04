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
