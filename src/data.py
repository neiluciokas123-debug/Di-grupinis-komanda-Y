"""Duomenų įkėlimas, valymas, anonimizavimas ir skaidymas į train/val/test.

Paleidimas:  python -m src.data
Rezultatas:  data/processed/{train,val,test}.csv
"""
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

# --- Valymo taisyklės --------------------------------------------------------
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_WIKI_USER = re.compile(r"\[\[\s*User(?: talk)?:[^\]|]*(\|[^\]]*)?\]\]", re.I)
_TIMESTAMP = re.compile(r"\d{1,2}:\d{2},? \d{1,2} \w+ \d{4} \(UTC\)")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Pašalina asmens duomenis (IP, el. paštus, URL, vartotojų vardus) ir triukšmą.

    Didžiosios raidės paliekamos: modeliai patys normalizuoja (TF-IDF lowercase,
    DistilBERT uncased), o sąsajoje originalus tekstas gražiau atrodo.
    """
    text = str(text)
    text = _URL.sub(" [URL] ", text)
    text = _EMAIL.sub(" [EMAIL] ", text)
    text = _IP.sub(" [IP] ", text)
    text = _WIKI_USER.sub(" [USER] ", text)
    text = _TIMESTAMP.sub(" ", text)
    text = _SPACES.sub(" ", text)
    return text.strip()


def find_raw_file() -> Path:
    for name in ("train.csv", "train.csv.zip"):
        p = C.DATA_RAW / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Nerastas {C.DATA_RAW / 'train.csv'}.\n"
        "Atsisiųskite duomenis iš https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data\n"
        "ir įdėkite train.csv į data/raw/ (žr. README.md)."
    )


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(find_raw_file())
    missing = [c for c in ["comment_text"] + C.CATEGORY_LABELS if c not in df.columns]
    if missing:
        raise ValueError(f"Faile trūksta stulpelių: {missing}")
    return df


def prepare(seed: int = C.SEED) -> None:
    df = load_raw()
    n0 = len(df)

    df["comment_text"] = df["comment_text"].fillna("").map(clean_text)
    df = df[df["comment_text"].str.len() > 0]
    df = df.drop_duplicates(subset="comment_text")
    df[C.MAIN_LABEL] = (df[C.CATEGORY_LABELS].sum(axis=1) > 0).astype(int)

    cols = ["id", "comment_text"] + C.ALL_LABELS if "id" in df.columns else ["comment_text"] + C.ALL_LABELS
    df = df[cols].reset_index(drop=True)

    # 80 / 10 / 10, stratifikuota pagal pagrindinę etiketę, kad visur būtų vienodas agresyvių dalis
    train, rest = train_test_split(df, test_size=0.2, stratify=df[C.MAIN_LABEL], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[C.MAIN_LABEL], random_state=seed)

    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(C.DATA_PROCESSED / f"{name}.csv", index=False)

    print(f"Įkelta: {n0}  ->  po valymo: {len(df)} (pašalinta tuščių/dublikatų: {n0 - len(df)})")
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"  {name:5s}: {len(part):7d} komentarų, agresyvių {part[C.MAIN_LABEL].mean():.1%}")
    print(f"Išsaugota: {C.DATA_PROCESSED}")


def load_split(name: str) -> pd.DataFrame:
    p = C.DATA_PROCESSED / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Nėra {p}. Pirma paleiskite: python -m src.data")
    df = pd.read_csv(p)
    df["comment_text"] = df["comment_text"].fillna("").astype(str)
    return df


if __name__ == "__main__":
    prepare()
