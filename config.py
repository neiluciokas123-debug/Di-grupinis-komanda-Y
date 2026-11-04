"""Bendri projekto nustatymai: keliai, etiketės, atsitiktinumo sėkla."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Keliai ---------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"            # čia įdėkite Kaggle train.csv
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
BASELINE_DIR = MODELS_DIR / "baseline"
DISTILBERT_DIR = MODELS_DIR / "distilbert"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# --- Etiketės --------------------------------------------------------------
# Originalios 6 Jigsaw kategorijos
CATEGORY_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# Pagrindinė (dvejetainė) užduotis: ar komentaras agresyvus (bent viena kategorija = 1)
MAIN_LABEL = "aggressive"
ALL_LABELS = [MAIN_LABEL] + CATEGORY_LABELS

# Kategorijų pavadinimai lietuviškai (sąsajai ir grafikams)
LABELS_LT = {
    "aggressive": "Agresyvus",
    "toxic": "Toksiškas",
    "severe_toxic": "Labai toksiškas",
    "obscene": "Nešvankus",
    "threat": "Grasinimas",
    "insult": "Įžeidimas",
    "identity_hate": "Neapykanta grupei",
}

# --- Moderavimo zonos (žmogus priima galutinį sprendimą) --------------------
# tikimybė < LOW  -> publikuoti
# LOW..HIGH       -> siųsti moderatoriui peržiūrai
# > HIGH          -> laikinai paslėpti, kol moderatorius patvirtins
ZONE_LOW = 0.40
ZONE_HIGH = 0.80

SEED = 42
DISTILBERT_BASE = "distilbert-base-uncased"
MAX_LEN = 128

for _d in (DATA_RAW, DATA_PROCESSED, BASELINE_DIR, DISTILBERT_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
