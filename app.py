"""Grafinė sąsaja (demonstracijai):  streamlit run app.py

Sistema yra INFORMACINĖ: ji tik siūlo sprendimą, o galutinį sprendimą (publikuoti ar
pašalinti) priima žmogus-moderatorius.
"""
import html
import time

import pandas as pd
import streamlit as st

import config as C
from src.explain import occlusion
from src.predict import available_models, zone

st.set_page_config(page_title="Komentarų agresijos filtras", page_icon="🛡️", layout="wide")

ZONES = {
    "publish": ("✅ Publikuoti", "#1e8449", "Komentaras atrodo neagresyvus."),
    "review": ("⚠️ Siųsti moderatoriui", "#b9770e", "Neaišku – reikia žmogaus sprendimo."),
    "hide": ("⛔ Laikinai paslėpti", "#c0392b", "Didelė agresijos tikimybė. Paslėpta, kol moderatorius patvirtins."),
}
EXAMPLES = [
    "Thanks for fixing the article, great work!",
    "This edit is wrong, please check the sources before reverting it again.",
    "You are an idiot and nobody wants you here.",
    "Shut up, you pathetic loser.",
    "I will find you and you will regret this.",
    "I am a proud gay man.",
]


@st.cache_resource(show_spinner="Įkeliamas modelis...")
def get_model(name: str):
    return available_models()[name]()


def highlight(weights) -> str:
    """Žodžiai nuspalvinami pagal okliuzijos svorį: raudona – didina agresyvumą."""
    if not weights:
        return ""
    vmax = max(abs(v) for _, v in weights) or 1.0
    spans = []
    for w, v in weights:
        a = min(abs(v) / vmax, 1.0) * 0.75
        color = f"rgba(192,57,43,{a:.2f})" if v > 0 else f"rgba(39,174,96,{a:.2f})"
        spans.append(f'<span title="{v:+.3f}" style="background:{color};padding:2px 3px;'
                     f'border-radius:4px;margin:1px;display:inline-block">{html.escape(w)}</span>')
    return '<div style="line-height:2.1;font-size:1.05rem">' + " ".join(spans) + "</div>"


def set_example(t):
    st.session_state["text"] = t


# ------------------------------------------------------------------ šoninė juosta
models = available_models()
st.sidebar.title("🛡️ Agresijos filtras")
if not models:
    st.error("Nėra ištreniruotų modelių. Paleiskite `python -m src.train_baseline` (žr. README.md).")
    st.stop()
model_name = st.sidebar.radio("Modelis", list(models))
model = get_model(model_name)
thr = model.thresholds[C.MAIN_LABEL]
st.sidebar.markdown(
    f"""
**Sprendimo zonos** (agresyvumo tikimybė):
- < {C.ZONE_LOW:.2f} → publikuoti
- {C.ZONE_LOW:.2f}–{C.ZONE_HIGH:.2f} → moderatoriui
- ≥ {C.ZONE_HIGH:.2f} → paslėpti iki patvirtinimo

Modelio F1-optimalus slenkstis: **{thr:.2f}**

---
Sistema **tik pataria**. Galutinį sprendimą visada priima žmogus.
Veikia tik **anglų kalba**.
"""
)

tab1, tab2, tab3 = st.tabs(["💬 Vienas komentaras", "📋 Komentarų srautas (CSV)", "📊 Modelių palyginimas"])

# ------------------------------------------------------------------ 1. vienas komentaras
with tab1:
    st.subheader("Įveskite komentarą (anglų kalba)")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        cols[i % 3].button(ex[:38] + ("…" if len(ex) > 38 else ""), key=f"ex{i}",
                           on_click=set_example, args=(ex,))
    text = st.text_area("Komentaras", key="text", height=110, placeholder="Type a comment...")

    if text.strip():
        t0 = time.perf_counter()
        probs = model.predict_proba([text])[0]
        ms = (time.perf_counter() - t0) * 1000
        p = float(probs[0])
        label, color, desc = ZONES[zone(p)]

        c1, c2, c3 = st.columns([2, 1, 1])
        c1.markdown(f'<div style="background:{color};color:white;padding:14px 18px;border-radius:10px;'
                    f'font-size:1.3rem;font-weight:600">{label}</div><div style="margin-top:6px">{desc}</div>',
                    unsafe_allow_html=True)
        c2.metric("Agresyvumo tikimybė", f"{p:.1%}")
        c3.metric("Apdorojimo laikas", f"{ms:.0f} ms")

        if model.has_categories:
            st.markdown("**Kategorijos**")
            cat = pd.DataFrame({"Kategorija": [C.LABELS_LT[l] for l in C.CATEGORY_LABELS],
                                "Tikimybė": probs[1:]}).set_index("Kategorija")
            st.bar_chart(cat, horizontal=True, height=230)

        st.markdown("**Kodėl? (XAI – kiekvieno žodžio įtaka)**  "
                    "<span style='color:#c0392b'>raudona</span> – didina agresyvumą, "
                    "<span style='color:#27ae60'>žalia</span> – mažina",
                    unsafe_allow_html=True)
        with st.spinner("Skaičiuojama..."):
            st.markdown(highlight(occlusion(model, text)), unsafe_allow_html=True)

# ------------------------------------------------------------------ 2. CSV srautas
with tab2:
    st.subheader("Paketinis komentarų tikrinimas")
    st.caption("Įkelkite CSV su stulpeliu `comment_text` (arba `text`). Moderatorius peržiūri "
               "pažymėtus komentarus ir priima galutinį sprendimą.")
    up = st.file_uploader("CSV failas", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        col = "comment_text" if "comment_text" in df.columns else ("text" if "text" in df.columns else None)
        if col is None:
            st.error("Nerastas stulpelis `comment_text` arba `text`.")
        else:
            df = df.head(5000)
            with st.spinner(f"Analizuojama {len(df)} komentarų..."):
                pr = model.predict_proba(df[col].fillna("").astype(str).tolist())
            out = pd.DataFrame({"Komentaras": df[col].astype(str), "Tikimybė": pr[:, 0].round(3)})
            out["Siūlymas"] = [ZONES[zone(x)][0] for x in out["Tikimybė"]]
            out["Moderatorius: pašalinti?"] = out["Tikimybė"] >= C.ZONE_HIGH
            out = out.sort_values("Tikimybė", ascending=False)

            k1, k2, k3 = st.columns(3)
            k1.metric("Publikuoti", int((out["Tikimybė"] < C.ZONE_LOW).sum()))
            k2.metric("Moderatoriui", int(out["Tikimybė"].between(C.ZONE_LOW, C.ZONE_HIGH, inclusive="left").sum()))
            k3.metric("Paslėpti", int((out["Tikimybė"] >= C.ZONE_HIGH).sum()))

            edited = st.data_editor(out, hide_index=True,
                                    disabled=["Komentaras", "Tikimybė", "Siūlymas"])
            st.download_button("⬇️ Atsisiųsti moderatoriaus sprendimus",
                               edited.to_csv(index=False).encode("utf-8"),
                               "moderavimo_sprendimai.csv", "text/csv")

# ------------------------------------------------------------------ 3. palyginimas
with tab3:
    st.subheader("Bazinis vs. patobulintas modelis (test aibė)")
    cmp = C.METRICS_DIR / "comparison.csv"
    if cmp.exists():
        st.dataframe(pd.read_csv(cmp).round(3), hide_index=True)
        lat = C.METRICS_DIR / "latency.csv"
        if lat.exists():
            st.markdown("**Inferencijos laikas (CPU, 1 komentaras)**")
            st.dataframe(pd.read_csv(lat), hide_index=True)
        for f, cap in [("metrics_comparison.png", "Metrikų palyginimas"),
                       ("confusion_matrices.png", "Konfuzijos matricos"),
                       ("roc_pr_curves.png", "ROC ir PR kreivės"),
                       ("categories_f1.png", "F1 pagal kategoriją"),
                       ("bias_test.png", "Šališkumo testas"),
                       ("distilbert_training.png", "DistilBERT treniravimas")]:
            if (C.FIGURES_DIR / f).exists():
                st.image(str(C.FIGURES_DIR / f), caption=cap)
    else:
        st.info("Rezultatų dar nėra. Paleiskite `python -m src.evaluate`.")
