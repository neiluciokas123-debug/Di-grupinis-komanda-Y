"""DI ŽODŽIŲ FILTRAS: paslepia tik tuos žodžius, kurie daro komentarą agresyvų.

Kaip veikia (godusis algoritmas, paremtas XAI okliuzija):
  1. Modelis įvertina komentaro agresyvumo tikimybę.
  2. Jei ji >= tikslo ribos, kiekvienam žodžiui apskaičiuojama jo įtaka
     (kiek nukrenta agresyvumas, kai žodis pašalinamas) – žr. src/explain.py.
  3. Įtakingiausias žodis pakeičiamas žvaigždutėmis (pvz. „idiot“ -> „*****“).
  4. Kartojama, kol komentaras tampa neagresyvus arba pasiekiamas žodžių limitas.

Skirtumas nuo juodojo sąrašo: sprendžia DI pagal kontekstą, todėl
„kill the process“ paliekama, o „I will kill you“ – ne; atpažįstami ir
užmaskuoti žodžiai (pvz. „idi0t“), kurių jokiame sąraše nėra.

Komandinė eilutė:
    python -m src.censor "You are a stupid idiot and nobody likes you"
    python -m src.censor --eval 300          # įvertinimas test aibėje
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from src.explain import occlusion

_WORD = re.compile(r"\S+")
_MASKABLE = re.compile(r"[A-Za-z0-9@$!|]")  # raidės ir dažni užmaskavimo simboliai

# Tarnybiniai žodžiai (you, are, the, my...) patys nieko neįžeidžia, nors modelis
# juos sieja su asmeniniais išpuoliais („you are ...“). Jų neslepiame, kad
# išvalytas komentaras liktų suprantamas.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
_KEEP = set(ENGLISH_STOP_WORDS) | {"im", "ur", "u", "dont", "cant", "lol", "like", "likes"}
# Tapatybės žodžiai nėra įžeidimai – jų slėpimas būtų diskriminacinis (žr. šališkumo testą
# src/evaluate.py). Modelis juos klaidingai sieja su agresija, todėl neleidžiame jų slėpti.
_KEEP |= {"gay", "lesbian", "transgender", "trans", "bisexual", "queer", "muslim", "muslims",
          "islam", "christian", "christians", "jewish", "jew", "jews", "atheist", "black",
          "blacks", "white", "whites", "asian", "african", "american", "woman", "women",
          "man", "men", "immigrant", "immigrants", "lithuanian", "disabled", "old", "young",
          "dog"}


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


@dataclass
class CensorResult:
    original: str
    censored: str
    prob_before: float
    prob_after: float
    masked_words: list = field(default_factory=list)
    success: bool = True  # ar pavyko pasiekti neagresyvų lygį

    @property
    def n_masked(self) -> int:
        return len(self.masked_words)


def _mask(word: str) -> str:
    """„idiot!!“ -> „*****!!“ – skyrybos ženklai paliekami, kad tekstas liktų skaitomas."""
    core = re.match(r"^(\W*)(.*?)(\W*)$", word)
    pre, mid, post = core.groups() if core else ("", word, "")
    if not mid:  # žodis vien iš simbolių
        return _MASKABLE.sub("*", word)
    return pre + "*" * len(mid) + post


def censor(predictor, text: str, target: float = C.ZONE_LOW, max_words: int = 8,
           min_effect: float = 0.7, strong_effect: float = 1.5) -> CensorResult:
    """Paslepia mažiausią žodžių kiekį, kad komentaro agresyvumas nukristų žemiau `target`.

    target     – siekiama agresyvumo tikimybė (numatytai moderavimo „publikuoti“ riba 0.40)
    max_words  – daugiausia kiek žodžių leidžiama paslėpti; jei neužtenka – komentaras
                 laikomas agresyviu iš esmės (success=False) ir paliekamas moderatoriui
    min_effect – mažiausia žodžio įtaka (log-odds), kad jis būtų slepiamas
    strong_effect – žodžiai su tokia ar didesne įtaka slepiami net tada, kai komentaras
                 jau pasiekė saugų lygį (kad neliktų pavienių įžeidimų)
    """
    words = _WORD.findall(text)
    p0 = float(predictor.predict_proba([text])[0, 0])
    if p0 < target or not words:
        return CensorResult(text, text, p0, p0, [], True)

    # Modeliui paduodame tekstą BE paslėptų žodžių: žvaigždutės pačios atrodo agresyviai,
    # nes duomenyse žmonės dažnai rašo „f***“. Vartotojas mato žvaigždutes.
    masked_idx: list[int] = []
    p = p0
    for _ in range(max_words):
        active = [i for i in range(len(words)) if i not in masked_idx]
        weights = occlusion(predictor, " ".join(words[i] for i in active), max_words=len(active))
        candidates = [(active[j], w) for j, (tok, w) in enumerate(weights)
                      if _norm(tok) and _norm(tok) not in _KEEP]
        if not candidates:
            break
        i, effect = max(candidates, key=lambda x: x[1])
        if effect < min_effect:
            break  # likę žodžiai patys savaime neagresyvūs – agresija slypi prasmėje
        if p < target and effect < strong_effect:
            break  # komentaras jau saugus, likę žodžiai – tik silpnai „įtartini“
        masked_idx.append(i)
        p = float(predictor.predict_proba([" ".join(w for k, w in enumerate(words)
                                                     if k not in masked_idx)])[0, 0])
    current = [(_mask(w) if k in masked_idx else w) for k, w in enumerate(words)]

    # Atkuriame originalų tarpų išdėstymą: keičiame tik paslėptus žodžius
    out, pos = [], 0
    for k, m in enumerate(_WORD.finditer(text)):
        out.append(text[pos:m.start()])
        out.append(current[k] if k in masked_idx else m.group())
        pos = m.end()
    out.append(text[pos:])
    censored = "".join(out)
    return CensorResult(text, censored, p0, p, [words[i] for i in sorted(masked_idx)], p < target)


def evaluate(predictor, n: int = 300) -> None:
    """Kiek agresyvių komentarų galima „išvalyti“ paslepiant kelis žodžius?"""
    import pandas as pd
    from src.data import load_split
    test = load_split("test")
    short = test[test["comment_text"].str.split().str.len().between(3, 60)]
    agg = short[short[C.MAIN_LABEL] == 1].sample(min(n, (short[C.MAIN_LABEL] == 1).sum()),
                                                 random_state=C.SEED)
    rows = []
    for t in agg["comment_text"]:
        r = censor(predictor, t)
        rows.append({"komentaras": t[:200], "isvalytas": r.censored[:200],
                     "p_pries": round(r.prob_before, 3), "p_po": round(r.prob_after, 3),
                     "pasleptu_zodziu": r.n_masked, "pavyko": r.success,
                     "zodziai": ", ".join(r.masked_words)})
    df = pd.DataFrame(rows)
    df.to_csv(C.METRICS_DIR / "censor_examples.csv", index=False)
    flagged = df[df["p_pries"] >= C.ZONE_LOW]
    ok = flagged[flagged["pavyko"]]
    summary = pd.DataFrame([{
        "Modelis": predictor.name,
        "Agresyvių komentarų (modelis pažymėjo)": len(flagged),
        "Išvalyta paslepiant žodžius, %": round(100 * len(ok) / max(len(flagged), 1), 1),
        "Vid. paslėptų žodžių": round(ok["pasleptu_zodziu"].mean(), 2) if len(ok) else np.nan,
        "Vid. agresyvumas prieš": round(flagged["p_pries"].mean(), 3),
        "Vid. agresyvumas po": round(flagged["p_po"].mean(), 3),
    }])
    summary.to_csv(C.METRICS_DIR / "censor_summary.csv", index=False)
    print(summary.T.to_string(header=False))
    print(f"Pavyzdžiai: {C.METRICS_DIR / 'censor_examples.csv'}")


if __name__ == "__main__":
    import argparse
    from src.predict import available_models
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--model", choices=["baseline", "distilbert"], default=None)
    ap.add_argument("--eval", type=int, default=0, help="Įvertinti su N test aibės komentarų")
    a = ap.parse_args()
    models = available_models()
    if not models:
        sys.exit("Nėra ištreniruotų modelių.")
    key = {"baseline": "TF-IDF + LogReg", "distilbert": "DistilBERT"}.get(a.model) or next(iter(models))
    pred = models[key]()
    if a.eval:
        evaluate(pred, a.eval)
    for t in a.texts:
        r = censor(pred, t)
        print(f"[{key}] {r.prob_before:.2f} -> {r.prob_after:.2f}  {r.censored}"
              + ("" if r.success else "   (nepavyko išvalyti – reikia moderatoriaus)"))
