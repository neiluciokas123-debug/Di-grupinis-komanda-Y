"""PERSPĖJIMAS PRIEŠ PASKELBIANT (tema Nr. 49).

Kai vartotojas spaudžia „Skelbti“, sistema patikrina jo komentarą ir, jei randa
toksiškumo požymių, PERSPĖJA autorių bei pasiūlo pataisyti. Galutinį sprendimą
(redaguoti ar skelbti vis tiek) visada priima pats autorius – sistema nieko neblokuoja.

Sujungiami du sluoksniai:
  1. DI modelis (TF-IDF+LogReg arba DistilBERT) – agresija, patyčios, įžeidimai, grasinimai;
  2. šauksmų detektorius (src/shouting.py) – DIDŽIOSIOS raidės, „!!!“, ištęsti žodžiai.

Perspėjimo lygiai:
  ok      – skelbiama iš karto;
  soft    – švelnus perspėjimas („Gal norėtumėte perfrazuoti?“): tik šauksmas
            arba agresyvumas 0.40–0.80;
  strong  – aiškus perspėjimas: agresyvumas >= 0.80 arba agresija (>= 0.40)
            kartu su stipriu šauksmu.

Komandinė eilutė:
    python -m src.warn "YOU ARE AN IDIOT!!!" "Thanks for the help"
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from src.shouting import ShoutResult, calm_down, detect

# Kurios modelio kategorijos rodomos kaip „patyčių“ požymiai
BULLYING_LABELS = ["insult", "threat", "identity_hate", "severe_toxic", "obscene"]
CATEGORY_MIN = 0.30          # kategorija paminima perspėjime, jei tikimybė >= 0.30


@dataclass
class Warning:
    text: str
    level: str                          # "ok" | "soft" | "strong"
    prob: float                         # agresyvumo tikimybė (DI modelis)
    shout: ShoutResult
    categories: list = field(default_factory=list)   # [(lietuviškas pavadinimas, tikimybė)]
    title: str = ""
    reasons: list = field(default_factory=list)
    calm_text: str | None = None        # pasiūlymas be šauksmų
    censor: object = None               # src.censor.CensorResult (jei skaičiuota)

    @property
    def needs_warning(self) -> bool:
        return self.level != "ok"


def check_before_post(predictor, text: str, with_censor: bool = True) -> Warning:
    """Patikrina komentarą prieš paskelbiant ir suformuoja perspėjimą autoriui."""
    probs = predictor.predict_proba([text])[0]
    p = float(probs[0])
    shout = detect(text)

    cats = []
    if getattr(predictor, "has_categories", False):
        for lab in BULLYING_LABELS:
            i = C.ALL_LABELS.index(lab)
            if probs[i] >= CATEGORY_MIN:
                cats.append((C.LABELS_LT[lab], float(probs[i])))
        cats.sort(key=lambda x: -x[1])

    reasons = []
    if p >= C.ZONE_LOW:
        reasons.append(f"DI modelis įvertino žinutę kaip galimai agresyvią ({p:.0%})")
    if cats:
        reasons.append("aptikti požymiai: " + ", ".join(f"{n.lower()} ({v:.0%})" for n, v in cats))
    reasons += shout.reasons

    if p >= C.ZONE_HIGH or (p >= C.ZONE_LOW and shout.level == "strong"):
        # agresija + stiprus šauksmas kartu („YOU IDIOT!!!“) -> aiškus perspėjimas
        level, title = "strong", "⛔ Jūsų žinutė gali įžeisti kitus"
    elif p >= C.ZONE_LOW or shout.is_shouting:
        level = "soft"
        title = ("📢 Atrodo, kad jūs šaukiate" if p < C.ZONE_LOW
                 else "⚠️ Jūsų žinutė gali būti suprasta kaip agresyvi")
    else:
        level, title = "ok", "✅ Žinutė atrodo draugiška"

    w = Warning(text, level, p, shout, cats, title, reasons)
    if shout.is_shouting:
        w.calm_text = calm_down(text)
    if with_censor and p >= C.ZONE_LOW:
        from src.censor import censor
        # žodžių filtrą taikome ramesnei versijai, kad pasiūlymas spręstų abi problemas
        w.censor = censor(predictor, w.calm_text or text)
    return w


if __name__ == "__main__":
    from src.predict import available_models
    models = available_models()
    if not models:
        sys.exit("Nėra ištreniruotų modelių. Žr. README.md")
    name = next(iter(models))
    pred = models[name]()
    for t in sys.argv[1:] or ["YOU ARE AN IDIOT!!!", "Thanks for the help", "WHY IS THIS LATE AGAIN???"]:
        w = check_before_post(pred, t)
        print(f"[{name}] {w.level:6s} {w.prob:.2f}  {t!r}\n   {w.title}")
        for r in w.reasons:
            print("   -", r)
        if w.calm_text:
            print("   be šauksmų:", w.calm_text)
        if w.censor is not None and w.censor.n_masked:
            print("   su paslėptais žodžiais:", w.censor.censored)
