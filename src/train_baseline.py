"""BAZINIS EKSPERIMENTAS (Baseline).

Treniruoja tris paprastus modelius, su kuriais lyginamas DistilBERT:
  0) Dažniausios klasės prognozė  – „nieko neišmokęs" modelis (visada „neagresyvus").
  1) Taisyklėmis grįstas filtras  – komentaras agresyvus, jei turi bent vieną žodį iš
                                    juodojo sąrašo; sąrašas sudaromas automatiškai iš
                                    train duomenų (src/eda.py santykinis dažnis).
  2) TF-IDF + LogisticRegression  – pagrindinis Baseline (žodžių 1-2 gramos +
                                    simbolių 2-5 gramos, class_weight="balanced").

Paleidimas:  python -m src.train_baseline
Rezultatas:  models/baseline/{tfidf_lr.joblib, rules.json}
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from src.data import load_split
from src.eda import relative_word_frequency
from src.metrics import best_threshold, binary_metrics
from src.text_utils import tokenize

# Hiperparametrai (aprašykite juos ataskaitos VI skyriuje)
WORD_PARAMS = dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=50_000,
                   sublinear_tf=True, strip_accents="unicode", lowercase=True)
CHAR_PARAMS = dict(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=50_000,
                   sublinear_tf=True, lowercase=True)
LR_PARAMS = dict(C=4.0, class_weight="balanced", max_iter=1000, solver="liblinear")


# --------------------------------------------------------------------------- #
def train_rules(train, val) -> dict:
    words = relative_word_frequency(train)
    candidates = words[words["ratio"] > 5]["word"].tolist()
    if not candidates:
        candidates = words["word"].head(50).tolist()
    val_tokens = [set(tokenize(t)) for t in val["comment_text"]]
    best = (-1.0, 10)
    for n in (10, 25, 50, 100, 200, 400):
        lex = set(candidates[:n])
        pred = [int(bool(tok & lex)) for tok in val_tokens]
        f1 = binary_metrics(val[C.MAIN_LABEL], pred, 0.5)["F1"]
        if f1 > best[0]:
            best = (f1, n)
    lexicon = candidates[: best[1]]
    print(f"[Taisyklės] žodyno dydis parinktas pagal val F1: {len(lexicon)} žodžių (val F1={best[0]:.3f})")
    return {"lexicon": lexicon}


def train_tfidf_lr(train, val) -> dict:
    t0 = time.time()
    word_vec = TfidfVectorizer(**WORD_PARAMS)
    char_vec = TfidfVectorizer(**CHAR_PARAMS)
    X_train = hstack([word_vec.fit_transform(train["comment_text"]),
                      char_vec.fit_transform(train["comment_text"])]).tocsr()
    X_val = hstack([word_vec.transform(val["comment_text"]),
                    char_vec.transform(val["comment_text"])]).tocsr()
    print(f"[TF-IDF] požymių: {X_train.shape[1]}, vektorizavimas {time.time() - t0:.0f} s")

    models, thresholds = {}, {}
    for label in C.ALL_LABELS:
        y = train[label].values
        if y.sum() == 0:            # labai mažuose bandomuosiuose rinkiniuose
            models[label] = None
            thresholds[label] = 0.5
            continue
        clf = LogisticRegression(**LR_PARAMS).fit(X_train, y)
        models[label] = clf
        thresholds[label] = best_threshold(val[label], clf.predict_proba(X_val)[:, 1])
        m = binary_metrics(val[label], clf.predict_proba(X_val)[:, 1], thresholds[label])
        print(f"  {label:14s} val F1={m['F1']:.3f}  ROC-AUC={m['ROC-AUC']:.3f}  slenkstis={thresholds[label]:.2f}")
    print(f"[TF-IDF + LR] treniravimas baigtas per {time.time() - t0:.0f} s")
    return {"word_vec": word_vec, "char_vec": char_vec, "models": models, "thresholds": thresholds}


def main() -> None:
    train, val = load_split("train"), load_split("val")
    print(f"Train: {len(train)}, Val: {len(val)}")

    rules = train_rules(train, val)
    with open(C.BASELINE_DIR / "rules.json", "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=1)

    bundle = train_tfidf_lr(train, val)
    joblib.dump(bundle, C.BASELINE_DIR / "tfidf_lr.joblib", compress=3)
    print(f"Išsaugota: {C.BASELINE_DIR}")


if __name__ == "__main__":
    np.random.seed(C.SEED)
    main()
