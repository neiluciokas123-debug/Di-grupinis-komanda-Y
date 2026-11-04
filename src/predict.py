"""Vieninga sąsaja visiems modeliams: predict_proba(texts) -> masyvas (n, len(ALL_LABELS)).

Naudoja evaluate.py, explain.py ir app.py.

Komandinė eilutė:
    python -m src.predict "You are an idiot" "Thanks for the help!"
    python -m src.predict --model baseline "some text"
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from src.data import clean_text
from src.text_utils import tokenize


class MajorityPredictor:
    name = "Dažniausia klasė"
    has_categories = False

    def __init__(self):
        self.thresholds = {l: 0.5 for l in C.ALL_LABELS}

    def predict_proba(self, texts):
        return np.zeros((len(texts), len(C.ALL_LABELS)))


class RulePredictor:
    name = "Taisyklės (žodynas)"
    has_categories = False

    def __init__(self, path=C.BASELINE_DIR / "rules.json"):
        with open(path, encoding="utf-8") as f:
            self.lexicon = set(json.load(f)["lexicon"])
        self.thresholds = {l: 0.5 for l in C.ALL_LABELS}

    def predict_proba(self, texts):
        hits = np.array([float(bool(set(tokenize(clean_text(t))) & self.lexicon)) for t in texts])
        out = np.zeros((len(texts), len(C.ALL_LABELS)))
        out[:, 0] = hits
        return out


class BaselinePredictor:
    name = "TF-IDF + LogReg"
    has_categories = True

    def __init__(self, path=C.BASELINE_DIR / "tfidf_lr.joblib"):
        b = joblib.load(path)
        self.word_vec, self.char_vec = b["word_vec"], b["char_vec"]
        self.models, self.thresholds = b["models"], b["thresholds"]

    def predict_proba(self, texts):
        texts = [clean_text(t) for t in texts]
        X = hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr()
        out = np.zeros((len(texts), len(C.ALL_LABELS)))
        for i, lab in enumerate(C.ALL_LABELS):
            if self.models.get(lab) is not None:
                out[:, i] = self.models[lab].predict_proba(X)[:, 1]
        return out

    def predict_logit(self, texts):
        """Agresyvumo log-odds (be sigmoidės) – neprisotinamas, naudojamas XAI okliuzijai."""
        texts = [clean_text(t) for t in texts]
        X = hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr()
        return self.models[C.MAIN_LABEL].decision_function(X)


class DistilBertPredictor:
    name = "DistilBERT"
    has_categories = True

    def __init__(self, path=C.DISTILBERT_DIR, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        with open(Path(path) / "thresholds.json") as f:
            self.thresholds = json.load(f)

    def _logits(self, texts, batch_size=64):
        torch = self.torch
        texts = [clean_text(t) for t in texts]
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=C.MAX_LEN,
                                     padding=True, return_tensors="pt").to(self.device)
                out.append(self.model(**enc).logits.float().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, len(C.ALL_LABELS)))

    def predict_proba(self, texts, batch_size=64):
        return 1 / (1 + np.exp(-self._logits(texts, batch_size)))

    def predict_logit(self, texts):
        """Agresyvumo log-odds (be sigmoidės) – naudojamas XAI okliuzijai."""
        return self._logits(texts)[:, 0]


def available_models() -> dict:
    """Grąžina tik tuos modelius, kurie jau ištreniruoti (failai egzistuoja)."""
    res = {}
    if (C.DISTILBERT_DIR / "thresholds.json").exists():
        try:
            import torch, transformers  # noqa: F401
            res["DistilBERT"] = DistilBertPredictor
        except ImportError:
            pass
    if (C.BASELINE_DIR / "tfidf_lr.joblib").exists():
        res["TF-IDF + LogReg"] = BaselinePredictor
    return res


def zone(prob: float) -> str:
    if prob >= C.ZONE_HIGH:
        return "hide"
    if prob >= C.ZONE_LOW:
        return "review"
    return "publish"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--model", choices=["baseline", "distilbert"], default=None)
    a = ap.parse_args()
    models = available_models()
    if not models:
        sys.exit("Nėra ištreniruotų modelių. Žr. README.md")
    key = {"baseline": "TF-IDF + LogReg", "distilbert": "DistilBERT"}.get(a.model) or next(iter(models))
    pred = models[key]()
    probs = pred.predict_proba(a.texts)
    for t, p in zip(a.texts, probs):
        cats = ", ".join(f"{C.LABELS_LT[l]} {p[i]:.2f}" for i, l in enumerate(C.ALL_LABELS[1:], 1) if p[i] > 0.3)
        print(f"[{key}] {p[0]:.3f}  {zone(p[0]).upper():8s} {t[:70]!r}  {cats}")
