"""ŠAUKSMŲ DETEKTORIUS: ar komentaras „rėkia“ (RAŠOMAS DIDŽIOSIOMIS, „!!!“, „STOOOP“).

Kodėl atskiras modulis? Abu DI modeliai tekstą paverčia mažosiomis raidėmis
(TF-IDF lowercase=True, DistilBERT *uncased*), todėl šauksmų jie NEMATO.
Šauksmas pats savaime nebūtinai įžeidžia („THANK YOU SO MUCH!!!“), todėl jis
vertinamas atskirai nuo agresijos ir sukelia tik švelnų perspėjimą.

Požymiai (paprastos, paaiškinamos taisyklės – žr. EDA: reports/metrics/eda_style.csv):
  * didžiųjų raidžių dalis tarp visų raidžių (kai raidžių >= 8);
  * žodžiai VIEN DIDŽIOSIOMIS (>= 3 raidžių, išskyrus santrumpas: USA, NATO, BBC...);
  * šauktukų / klaustukų serijos („!!!“, „?!?!“);
  * ištęstos raidės („nooooo“, „STOOOP“).

Komandinė eilutė:
    python -m src.shouting "WHY DO YOU NEVER LISTEN!!!" "Thanks, NATO is in the USA news"
"""
import re
from dataclasses import dataclass, field

# Santrumpos ir anonimizavimo žymės nėra šauksmas
ACRONYMS = {
    "I", "OK", "USA", "UK", "EU", "UN", "NATO", "BBC", "CNN", "TV", "PC", "FBI", "CIA", "CEO",
    "AI", "NASA", "URL", "IP", "EMAIL", "USER", "ID", "PDF", "FAQ", "GPS", "DNA", "HIV", "NFL",
    "NBA", "UEFA", "FIFA", "LOL", "OMG", "BTW", "IMO", "IMHO", "WP", "NPOV", "POV", "ASAP",
    "HTML", "CSS", "API", "USB", "CPU", "GPU", "RAM", "DVD", "CD", "UTC", "GMT", "AM", "PM",
}

_LETTER = re.compile(r"[A-Za-z]")
_UPPER = re.compile(r"[A-Z]")
_WORD = re.compile(r"[A-Za-z']+")
_PUNCT_RUN = re.compile(r"[!?]{2,}")
_STRETCH = re.compile(r"([A-Za-z])\1{2,}")

# Slenksčiai (parinkti pagal EDA ir rankinius pavyzdžius; žr. README „Šauksmų detektorius“)
MIN_LETTERS = 8          # trumpiems tekstams („OK“, „NO“) didžiųjų raidžių dalis neskaičiuojama
CAPS_STRONG = 0.70       # >= 70 % raidžių didžiosios -> stiprus šauksmas
CAPS_MILD = 0.40
CAPS_WORDS_MILD = 2      # >= 2 žodžiai vien didžiosiomis
PUNCT_RUN_MILD = 3       # „!!!“ (3 ir daugiau ženklų iš eilės)
EXCL_TOTAL_MILD = 4      # arba iš viso >= 4 šauktukai


@dataclass
class ShoutResult:
    text: str
    level: str                      # "none" | "mild" | "strong"
    caps_ratio: float
    caps_words: list = field(default_factory=list)
    punct_runs: list = field(default_factory=list)
    stretched: list = field(default_factory=list)
    reasons: list = field(default_factory=list)   # lietuviški paaiškinimai vartotojui

    @property
    def is_shouting(self) -> bool:
        return self.level != "none"


def detect(text: str) -> ShoutResult:
    """Įvertina, ar tekstas „rėkia“. Grąžina lygį ir priežastis (paaiškinamumui)."""
    text = str(text)
    # santrumpos (USA, NATO...) neįskaičiuojamos į didžiųjų raidžių dalį
    no_acr = _WORD.sub(lambda m: "" if m.group(0).strip("'") in ACRONYMS else m.group(0), text)
    letters = len(_LETTER.findall(no_acr))
    caps_ratio = len(_UPPER.findall(no_acr)) / letters if letters else 0.0
    caps_words = [w for w in _WORD.findall(text)
                  if len(w.strip("'")) >= 3 and w.isupper() and w.strip("'") not in ACRONYMS]
    punct_runs = _PUNCT_RUN.findall(text)
    long_runs = [r for r in punct_runs if len(r) >= PUNCT_RUN_MILD]
    n_excl = text.count("!")
    stretched = [w for w in _WORD.findall(text)
                 if _STRETCH.search(w) and not w.lower().startswith("www")]

    reasons, points = [], 0
    if letters >= MIN_LETTERS and caps_ratio >= CAPS_STRONG:
        reasons.append(f"beveik visas tekstas parašytas DIDŽIOSIOMIS raidėmis ({caps_ratio:.0%})")
        points += 2
    elif (letters >= MIN_LETTERS and caps_ratio >= CAPS_MILD) or len(caps_words) >= CAPS_WORDS_MILD:
        reasons.append("keli žodžiai parašyti DIDŽIOSIOMIS: " + ", ".join(caps_words[:5]))
        points += 1
    if long_runs or n_excl >= EXCL_TOTAL_MILD:
        shown = ", ".join(dict.fromkeys(long_runs)) or f"{n_excl} šauktukai"
        reasons.append(f"daug šauktukų / klaustukų ({shown})")
        points += 1
    if stretched and (points or len(stretched) >= 2):
        # ištęstos raidės vienos („sooo cute“) – ne šauksmas, bet kartu su kitais požymiais – taip
        reasons.append("ištęsti žodžiai: " + ", ".join(stretched[:3]))
        points += 1

    level = "strong" if points >= 2 else ("mild" if points == 1 else "none")
    return ShoutResult(text, level, round(caps_ratio, 3), caps_words, punct_runs, stretched, reasons)


def calm_down(text: str) -> str:
    """Pasiūlo ramesnę versiją: mažosios raidės, vienas šauktukas, be ištęstų raidžių.

    „WHY DO YOU NEVER LISTEN!!! STOOOP“ -> „Why do you never listen! Stop“
    """
    def lower_word(m):
        w = m.group(0)
        return w if (not w.isupper() or w.strip("'") in ACRONYMS) else w.lower()

    out = _STRETCH.sub(lambda m: m.group(1), text)              # STOOOP -> STOP
    out = _WORD.sub(lower_word, out)                            # STOP -> stop
    out = re.sub(r"!+[!?]*|\?+[!?]*", lambda m: m.group(0)[0], out)  # !!! -> !, ?!?! -> ?
    out = re.sub(r"\bi\b", "I", out)                            # angliškas „I“
    # sakinio pradžia – didžioji raidė
    out = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["WHY DO YOU NEVER LISTEN!!! STOOOP", "Thanks for the help!",
                               "I read it on BBC and NATO news", "THANK YOU SO MUCH!!!"]:
        r = detect(t)
        print(f"{r.level:6s} caps={r.caps_ratio:.2f}  {t!r}")
        for x in r.reasons:
            print("         -", x)
        if r.is_shouting:
            print("         ramiau:", calm_down(t))
