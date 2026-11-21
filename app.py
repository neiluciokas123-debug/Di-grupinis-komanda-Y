"""Grafinė sąsaja (demonstracijai):  streamlit run app.py

Tema Nr. 49: prieš paskelbiant viešą komentarą sistema PERSPĖJA autorių, jei jo žinutė
turi toksiškumo požymių (patyčios, įžeidimai, šauksmai), ir pasiūlo pataisytą versiją.

Sistema yra INFORMACINĖ: ji nieko neblokuoja. Galutinį sprendimą (redaguoti ar skelbti
vis tiek) priima pats autorius; papildomas skirtukas skirtas moderatoriui.
"""
import html
import time

import pandas as pd
import streamlit as st

import config as C
from src.censor import censor
from src.explain import occlusion
from src.predict import available_models, zone
from src.shouting import detect as detect_shouting
from src.warn import check_before_post

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
    "Please kill the process before restarting the server.",
    "This article is crap, whoever wrote it is a moron.",
    "you are such an idi0t lol",
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


def set_draft(t):
    st.session_state["draft"] = t
    st.session_state["pending"] = None


# --------------------------------------------- būsena: paskelbti komentarai ir statistika
st.session_state.setdefault("feed", [])          # [(tekstas, kaip paskelbta)]
st.session_state.setdefault("pending", None)     # src.warn.Warning, laukiantis autoriaus sprendimo
st.session_state.setdefault("stats", {"Patikrinta": 0, "Perspėta": 0, "Pataisė": 0,
                                      "Skelbė vis tiek": 0})


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
- < {C.ZONE_LOW:.2f} → skelbiama iš karto
- {C.ZONE_LOW:.2f}–{C.ZONE_HIGH:.2f} → švelnus perspėjimas autoriui
- ≥ {C.ZONE_HIGH:.2f} → aiškus perspėjimas autoriui

**📢 Šauksmai** (DIDŽIOSIOS raidės, „!!!“, „STOOOP“) aptinkami
atskiru detektoriumi – DI modeliai raidžių dydžio nemato.

Modelio F1-optimalus slenkstis: **{thr:.2f}**

---
**🧹 DI žodžių filtras** paslepia tik tuos žodžius, kurie daro
komentarą agresyvų (pagal kontekstą, ne pagal sąrašą).

---
Sistema **tik perspėja**. Galutinį sprendimą visada priima žmogus.
Veikia tik **anglų kalba**.
"""
)

tab0, tab1, tab2, tab3 = st.tabs(["✍️ Rašyti komentarą", "🔍 Detali analizė",
                                  "📋 Moderatoriui (CSV)", "📊 Modelių palyginimas"])

# ------------------------------------------------------------------ 0. perspėjimas autoriui
WARN_STYLE = {"strong": ("#c0392b", "#fdecea"), "soft": ("#b9770e", "#fef5e7")}
AUTHOR_EXAMPLES = [
    "Great explanation, thanks for sharing!",
    "WHY DOES NOBODY READ THE RULES??? THIS IS RIDICULOUS!!!",
    "You are an idiot and nobody wants you here.",
    "This article is crap, whoever wrote it is a moron.",
    "SHUT UP, YOU PATHETIC LOSER!!!",
    "I disagree, the second source says the opposite.",
]


def publish(text: str, how: str) -> None:
    st.session_state["feed"].insert(0, (text, how))
    st.session_state["draft"] = ""
    st.session_state["pending"] = None


def on_post():
    text = st.session_state.get("draft", "").strip()
    if not text:
        return
    w = check_before_post(model, text)
    st.session_state["stats"]["Patikrinta"] += 1
    if w.needs_warning:
        st.session_state["stats"]["Perspėta"] += 1
        st.session_state["pending"] = w
    else:
        publish(text, "✅ be perspėjimo")


def on_accept_suggestion(text: str):
    st.session_state["stats"]["Pataisė"] += 1
    publish(text, "✏️ pataisyta pagal pasiūlymą")


def on_post_anyway():
    w = st.session_state["pending"]
    st.session_state["stats"]["Skelbė vis tiek"] += 1
    how = "⚠️ autorius skelbė nepaisydamas perspėjimo"
    if w.level == "strong":
        how += " (pažymėta moderatoriui)"
    publish(w.text, how)


def on_edit():
    st.session_state["stats"]["Pataisė"] += 1
    st.session_state["pending"] = None


with tab0:
    st.subheader("Parašykite viešą komentarą")
    st.caption("Prieš paskelbiant DI patikrina, ar žinutėje nėra patyčių, įžeidimų ar šauksmų, "
               "ir jei reikia – perspėja. Sprendžiate jūs: pataisyti ar skelbti vis tiek. "
               "Veikia tik anglų kalba.")
    cols = st.columns(3)
    for i, ex in enumerate(AUTHOR_EXAMPLES):
        cols[i % 3].button(ex[:38] + ("…" if len(ex) > 38 else ""), key=f"aex{i}",
                           on_click=set_draft, args=(ex,))
    st.text_area("Jūsų komentaras", key="draft", height=110, placeholder="Write a comment...")
    st.button("📤 Skelbti", type="primary", on_click=on_post)

    w = st.session_state["pending"]
    if w is not None and w.text == st.session_state.get("draft", "").strip():
        border, bg = WARN_STYLE[w.level]
        items = "".join(f"<li>{html.escape(r)}</li>" for r in w.reasons)
        hint = ("Tokia žinutė gali įskaudinti kitus ir būti paslėpta. Ar tikrai norite ją skelbti?"
                if w.level == "strong" else "Gal norėtumėte perfrazuoti? Rami žinutė dažniau sulaukia atsakymo.")
        st.markdown(f'<div style="border-left:6px solid {border};background:{bg};color:#222;'
                    f'padding:14px 18px;border-radius:8px;margin-top:8px">'
                    f'<div style="font-size:1.2rem;font-weight:600">{w.title}</div>'
                    f'<div style="margin:6px 0">{hint}</div><ul style="margin:0">{items}</ul></div>',
                    unsafe_allow_html=True)

        # Pasiūlymas: pirmiausia – paslėpti agresyvūs žodžiai (jei to užtenka), kitaip – be šauksmų
        suggestion = None
        if w.censor is not None and w.censor.n_masked and w.censor.success:
            suggestion = ("🧹 Pasiūlymas – paslėpti įžeidžiantys žodžiai", w.censor.censored)
        elif w.calm_text and w.prob < C.ZONE_LOW:
            suggestion = ("🔉 Pasiūlymas – ta pati mintis be šauksmų", w.calm_text)
        if suggestion:
            st.markdown(f"**{suggestion[0]}:**")
            st.markdown(f'<div style="border:1px solid #888;border-radius:8px;padding:10px 14px;'
                        f'font-family:monospace">{html.escape(suggestion[1])}</div>',
                        unsafe_allow_html=True)
        elif w.prob >= C.ZONE_LOW:
            st.info("DI nerado atskirų įžeidžiančių žodžių – agresija slypi pačioje sakinio prasmėje "
                    "(pvz. grasinimas ar pajuoka). Siūlome žinutę perrašyti savais žodžiais.")

        b1, b2, b3 = st.columns(3)
        if suggestion:
            b1.button("✅ Skelbti pasiūlytą versiją", on_click=on_accept_suggestion,
                      args=(suggestion[1],), use_container_width=True)
        b2.button("✏️ Redaguoti pačiam", on_click=on_edit, use_container_width=True)
        b3.button("📢 Skelbti vis tiek", on_click=on_post_anyway, use_container_width=True)

    if st.session_state["feed"]:
        st.markdown("---\n#### 💬 Paskelbti komentarai")
        for text, how in st.session_state["feed"][:20]:
            st.markdown(f'<div style="border:1px solid #ccc;border-radius:8px;padding:8px 12px;margin:6px 0">'
                        f'{html.escape(text)}<div style="font-size:0.8rem;color:#888;margin-top:4px">'
                        f'{html.escape(how)}</div></div>', unsafe_allow_html=True)

    s = st.session_state["stats"]
    if s["Patikrinta"]:
        st.markdown("---\n**Šios sesijos statistika** (vartotojo elgsena po perspėjimo)")
        k = st.columns(4)
        for col, (name, val) in zip(k, s.items()):
            col.metric(name, val)

# ------------------------------------------------------------------ 1. vienas komentaras
with tab1:
    st.subheader("Detali komentaro analizė (anglų kalba)")
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

        sh = detect_shouting(text)
        if sh.is_shouting:
            st.warning(f"📢 Šauksmas ({'stiprus' if sh.level == 'strong' else 'švelnus'}): "
                       + "; ".join(sh.reasons))

        if model.has_categories:
            st.markdown("**Kategorijos**")
            cat = pd.DataFrame({"Kategorija": [C.LABELS_LT[l] for l in C.CATEGORY_LABELS],
                                "Tikimybė": probs[1:]}).set_index("Kategorija")
            st.bar_chart(cat, horizontal=True, height=230)

        # ---- DI žodžių filtras
        st.markdown("### 🧹 DI žodžių filtras")
        if p < C.ZONE_LOW:
            st.info("Komentaras neagresyvus – valyti nereikia.")
        else:
            with st.spinner("DI ieško agresyvių žodžių..."):
                r = censor(model, text)
            st.markdown(f'<div style="border:1px solid #888;border-radius:10px;padding:12px 16px;'
                        f'font-size:1.15rem;font-family:monospace">{html.escape(r.censored)}</div>',
                        unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            f1.metric("Paslėpta žodžių", r.n_masked)
            f2.metric("Agresyvumas po valymo", f"{r.prob_after:.1%}",
                      delta=f"{(r.prob_after - r.prob_before) * 100:+.1f} p. p.", delta_color="inverse")
            if r.success and r.n_masked:
                st.success("✅ Išvalytą versiją galima publikuoti – agresija buvo tik atskiruose žodžiuose.")
            elif r.n_masked:
                st.warning("⚠️ Paslėpus žodžius komentaras vis tiek agresyvus – agresija slypi sakinio "
                           "prasmėje (pvz. grasinimas). Sprendžia moderatorius.")
            else:
                st.warning("⚠️ DI nerado atskirų agresyvių žodžių – agresija slypi sakinio prasmėje. "
                           "Sprendžia moderatorius.")

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
            if st.checkbox("🧹 Išvalyti agresyvius komentarus DI žodžių filtru (lėčiau)"):
                flagged = out.index[out["Tikimybė"] >= C.ZONE_LOW][:300]
                bar = st.progress(0.0, text="DI slepia agresyvius žodžius...")
                cleaned = {}
                for k, idx in enumerate(flagged):
                    cleaned[idx] = censor(model, out.at[idx, "Komentaras"])
                    bar.progress((k + 1) / max(len(flagged), 1))
                bar.empty()
                out["Išvalytas komentaras"] = [cleaned[i].censored if i in cleaned else c
                                               for i, c in zip(out.index, out["Komentaras"])]
                out["Agresyvumas po valymo"] = [round(cleaned[i].prob_after, 3) if i in cleaned else t
                                                for i, t in zip(out.index, out["Tikimybė"])]
                if len(flagged) == 300:
                    st.caption("Išvalyti pirmi 300 pažymėtų komentarų.")
            out = out.sort_values("Tikimybė", ascending=False)

            k1, k2, k3 = st.columns(3)
            k1.metric("Publikuoti", int((out["Tikimybė"] < C.ZONE_LOW).sum()))
            k2.metric("Moderatoriui", int(out["Tikimybė"].between(C.ZONE_LOW, C.ZONE_HIGH, inclusive="left").sum()))
            k3.metric("Paslėpti", int((out["Tikimybė"] >= C.ZONE_HIGH).sum()))

            edited = st.data_editor(out, hide_index=True,
                                    disabled=[c for c in out.columns if c != "Moderatorius: pašalinti?"])
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
