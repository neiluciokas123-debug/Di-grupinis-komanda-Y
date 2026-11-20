# 🛡️ Interneto komentarų agresijos filtras

Dirbtinio intelekto sistema, kuri įvertina, ar angliškas interneto komentaras yra agresyvus
(toksiškas, įžeidžiantis, grasinantis ir pan.), ir **pasiūlo** moderatoriui sprendimą:
publikuoti, peržiūrėti arba laikinai paslėpti. Galutinį sprendimą visada priima žmogus.

Lyginami du eksperimentai (pagal metodikos reikalavimą):

| | Metodas | Paskirtis |
|---|---|---|
| **Baseline** | TF-IDF (žodžių + simbolių n-gramos) + LogisticRegression | Paprastas, greitas, lengvai paaiškinamas |
| **Proposed** | DistilBERT (perkeliamasis mokymas) | Supranta kontekstą, atitinka < 200 ms reikalavimą |
| *Papildomi atskaitos taškai* | Dažniausios klasės prognozė; taisyklės (juodasis žodžių sąrašas) | Įrodo, kad DI pranašesnis už elementarų sprendimą |

---

## 1. Projekto klasifikavimas

| Kriterijus | Atsakymas | Poveikis metodikai |
|---|---|---|
| Duomenų pobūdis | **Tekstas (NLP)** | scikit-learn, HuggingFace `transformers`, PyTorch |
| Duomenų šaltinis | **Atviras duomenų rinkinys** (Jigsaw Toxic Comment, Kaggle) | Realių naujų asmens duomenų nerenkame; IP, el. paštai, URL ir vartotojų vardai pašalinami valant |
| Realus laikas | **Taip** (< 200 ms vienam komentarui) | Renkamės lengvą **DistilBERT**, ne BERT-large/RoBERTa-large; latencija matuojama (`latency.csv`) |
| Aparatūra | **Nėra** (grynai programinis) | – |
| Sprendimo kritiškumas | **Informacinis** (pataria) | Modelis nieko nešalina pats: tik siūlo zoną, moderatorius patvirtina |

## 2. Sistemos architektūra

```
                 ┌──────────────┐
 Komentaras ───▶ │  Valymas ir  │  URL/IP/el. paštas/vartotojas → [URL]/[IP]/[EMAIL]/[USER]
 (anglų k.)      │ anonimizavimas│
                 └──────┬───────┘
                        ▼
        ┌───────────────┴────────────────┐
        ▼                                ▼
 ┌──────────────┐                ┌──────────────┐
 │ TF-IDF + LR  │  (Baseline)    │  DistilBERT  │  (Proposed)
 └──────┬───────┘                └──────┬───────┘
        └───────────────┬───────────────┘
                        ▼
         7 tikimybės: agresyvus + 6 kategorijos
                        ▼
     ┌──────────────────┼────────────────────┐
     ▼                  ▼                    ▼
 < 0.40 Publikuoti  0.40–0.80 Moderatoriui  ≥ 0.80 Paslėpti iki patvirtinimo
                        │
                        ▼
          XAI: kurie žodžiai lėmė sprendimą
                        ▼
              👤 Moderatoriaus sprendimas
```

## 3. Failų struktūra

```
agresijos-filtras/
├── app.py                   # Streamlit sąsaja (demonstracijai)
├── config.py                # keliai, etiketės, moderavimo zonos
├── requirements.txt
├── Dockerfile
├── data/raw/                # čia dedamas Kaggle train.csv (žr. data/raw/README.md)
├── notebooks/
│   └── colab_treniravimas.ipynb   # visas eksperimentas Google Colab su GPU
├── src/
│   ├── data.py              # valymas, anonimizavimas, train/val/test 80/10/10
│   ├── eda.py               # histogramos, disbalansas, koreliacijos, „toksiški“ žodžiai
│   ├── train_baseline.py    # Baseline + taisyklių modelis
│   ├── train_distilbert.py  # Proposed (DistilBERT)
│   ├── evaluate.py          # lyginamoji lentelė, konfuzijos matricos, latencija, šališkumo testas
│   ├── explain.py           # XAI: okliuzija, LIME, globalūs LR koeficientai
│   ├── censor.py            # DI žodžių filtras: paslepia tik agresyvius žodžius
│   ├── predict.py           # vieninga modelių sąsaja + komandinė eilutė
│   ├── metrics.py
│   └── text_utils.py
└── reports/                 # sugeneruojama: figures/, metrics/, xai/
```

## 4. Paleidimas

### 4.1. Įdiegimas

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2. Duomenys

Atsisiųskite `train.csv` iš [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
ir įdėkite į `data/raw/` (detaliau: [data/raw/README.md](data/raw/README.md)).

### 4.3. Visas eksperimentas

```bash
python -m src.data               # 1. valymas ir skaidymas         (~1 min)
python -m src.eda                # 2. duomenų analizė, grafikai     (~1 min)
python -m src.train_baseline     # 3. Baseline                      (~5–10 min CPU)
python -m src.train_distilbert   # 4. DistilBERT                    (~20 min Colab T4 GPU)
python -m src.evaluate           # 5. palyginimas                   (~3 min GPU)
python -m src.explain            # 6. paaiškinamumas (XAI)
streamlit run app.py             # 7. demonstracija → http://localhost:8501
```

> **Be GPU?** DistilBERT treniruokite Google Colab: atidarykite `notebooks/colab_treniravimas.ipynb`,
> pasirinkite *Runtime → T4 GPU* ir paleiskite visus langelius. Pabaigoje atsisiųskite `rezultatai.zip`
> ir išarchyvuokite į projekto katalogą – tada `streamlit run app.py` veiks ir jūsų kompiuteryje (CPU).
>
> Greitas bandymas CPU: `python -m src.train_distilbert --max-train 5000 --epochs 1`

DI žodžių filtras komandinėje eilutėje:

```bash
python -m src.censor "This article is crap, whoever wrote it is a moron."
# -> This article is ****, whoever wrote it is a *****.
python -m src.censor --eval 300      # įvertinimas test aibėje -> reports/metrics/censor_summary.csv
```

Vieno komentaro patikrinimas komandinėje eilutėje:

```bash
python -m src.predict "You are an idiot" "Thanks for the help!"
```

Docker (su jau ištreniruotais modeliais `models/` kataloge):

```bash
docker build -t agresijos-filtras .
docker run -p 8501:8501 agresijos-filtras
```

## 5. Metodika

**Duomenys.** Jigsaw Toxic Comment Classification: ~159 tūkst. Vikipedijos diskusijų komentarų,
kuriuos anotavo žmonės pagal 6 kategorijas. Pagrindinė etiketė `aggressive` = 1, jei komentaras
turi bent vieną kategoriją (~10 % komentarų). Skaidymas 80/10/10, stratifikuotas.

**Disbalansas.** Agresyvių komentarų tik ~10 %, todėl:
- Baseline naudoja `class_weight="balanced"`;
- abiem modeliams sprendimo slenkstis parenkamas **val aibėje** maksimizuojant F1 (ne fiksuotas 0.5);
- DistilBERT galima treniruoti su `--pos-weight` (švelnus retų klasių svėrimas);
- vertinama **F1, Precision, Recall, PR-AUC**, nes Accuracy disbalanso atveju klaidina
  (modelis „visada neagresyvus“ gauna ~90 % Accuracy, bet F1 = 0).

**Baseline hiperparametrai:** žodžių 1–2 gramos (50 tūkst.), simbolių 2–5 gramos (50 tūkst.),
`sublinear_tf`, LogisticRegression `C=4`, `liblinear`.
Simbolių n-gramos padeda atpažinti užmaskuotus keiksmažodžius (pvz., „f*ck“, „idi0t“).

**DistilBERT hiperparametrai:** `distilbert-base-uncased`, maks. 128 žetonai, AdamW,
lr = 3e-5, batch 32, 2 epochos, linear warmup 6 %, fp16 (GPU), išsaugoma geriausia epocha pagal val F1.

**Paaiškinamumas (XAI):**
- *okliuzija* – po vieną pašalinamas žodis ir matuojama, kaip keičiasi agresyvumo log-odds
  (rodoma sąsajoje spalvomis);
- *LIME* – HTML paaiškinimai `reports/xai/`;
- *globalūs LR koeficientai* – kokie žodžiai Baseline modeliui reiškia agresiją.

**DI žodžių filtras (`src/censor.py`).** Užuot pašalinus visą agresyvų komentarą,
sistema gali paslėpti tik tuos žodžius, kurie jį daro agresyvų:

> *This article is crap, whoever wrote it is a moron.* → *This article is \*\*\*\*, whoever wrote it is a \*\*\*\*\*.*

Algoritmas (godusis, paremtas XAI okliuzija):
1. modelis įvertina komentarą; jei agresyvumas < 0.40 – nieko nedaroma;
2. kiekvienam žodžiui apskaičiuojama jo įtaka agresyvumui (log-odds pokytis jį pašalinus);
3. įtakingiausias žodis paslepiamas žvaigždutėmis, ir viskas kartojama,
   kol komentaras tampa neagresyvus (< 0.40) arba paslėpta 8 žodžiai;
4. jei paslėpus žodžius komentaras vis tiek agresyvus (pvz. grasinimas be keiksmažodžių),
   sprendimą priima moderatorius.

Skirtumai nuo juodojo sąrašo: sprendžia DI pagal kontekstą („kill the process“ paliekama,
„I will kill you“ – ne), atpažįstami užmaskuoti žodžiai („idi0t“).
Saugikliai: tarnybiniai žodžiai (you, are, the…) ir **tapatybės žodžiai** (gay, black,
muslim…) niekada neslepiami – jų slėpimas būtų diskriminacinis. Modeliui vertinti
paduodamas tekstas be paslėptų žodžių, nes pačios žvaigždutės („f***“) duomenyse siejasi su agresija.

| Modelis | Agresyvių komentarų | Išvalyta paslepiant žodžius | Vid. paslėptų žodžių | Agresyvumas prieš → po |
|---|---|---|---|---|
| TF-IDF + LogReg | 260 | 65,0 % | 2,1 | 0,92 → 0,38 |
| DistilBERT | | | | |

*(300 atsitiktinių agresyvių test komentarų, 3–60 žodžių. DistilBERT eilutė užpildoma paleidus
`python -m src.censor --model distilbert --eval 300`.)*
Baseline nepavykusių atvejų dažna priežastis – jis frazę „you are …“ laiko agresyvia net
be įžeidimų (pvz. „You are a nice person“ → 0,97), todėl tokie komentarai paliekami moderatoriui.

## 6. Rezultatai

Test aibė: 15 920 komentarų (agresyvių 10,2 %), klasė „agresyvus“. Gauta `python -m src.evaluate`
(Google Colab, T4 GPU; DistilBERT treniruotas su 50 000 komentarų, 2 epochos):

| Modelis | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Dažniausia klasė | 0.898 | 0.000 | 0.000 | 0.000 | 0.500 | 0.102 |
| Taisyklės (žodynas) | 0.943 | 0.742 | 0.677 | 0.708 | 0.825 | 0.535 |
| TF-IDF + LogReg (Baseline) | 0.963 | 0.854 | 0.766 | 0.807 | 0.980 | 0.899 |
| **DistilBERT (Proposed)** | **0.970** | **0.876** | **0.815** | **0.845** | **0.987** | **0.925** |

Inferencijos laikas (CPU, 1 komentaras): Baseline ~3 ms, DistilBERT ~80 ms (95 % – iki ~141 ms) – atitinka < 200 ms reikalavimą.
Šališkumo testas: neutralus sakinys su „gay“ – Baseline 0,96, DistilBERT 0,53; su „black“ – 0,57 ir 0,22.

Grafikai ataskaitai: `reports/figures/` (konfuzijos matricos, ROC/PR kreivės, treniravimo kreivės,
F1 pagal kategoriją, šališkumo testas, EDA). Sėkmės / nesėkmės pavyzdžiai: `reports/metrics/examples_*.csv`.

## 7. Etika, sauga ir privatumas

- **Privatumas:** naudojamas viešas, jau anotuotas duomenų rinkinys; nauji asmens duomenys nerenkami.
  Valant pašalinami IP adresai, el. paštai, URL ir Vikipedijos vartotojų vardai.
  Duomenys į repozitoriją nekeliami (Kaggle licencija), kiekvienas atsisiunčia pats.
- **Žmogus sprendimo cikle:** sistema tik siūlo. Net „Paslėpti“ zona reiškia laikiną paslėpimą,
  kol moderatorius patvirtins, o ne automatinį pašalinimą. Taip mažinama cenzūros rizika.
- **Šališkumas:** Jigsaw duomenyse tapatybės žodžiai („gay“, „muslim“, „black“) dažnai pasitaiko
  toksiškuose komentaruose, todėl modelis gali klaidingai žymėti neutralius sakinius.
  `evaluate.py` atlieka **šališkumo testą** su neutraliais šablonais (`bias_test.png`).
- **Saugumas:** modelis nieko negeneruoja, tik klasifikuoja. Nenaudojami komerciniai uždari API:
  visi modeliai veikia lokaliai (offline).

## 8. Ribotumai

- Veikia **tik anglų kalba**; lietuviški komentarai bus vertinami nepatikimai.
- Duomenys yra iš Vikipedijos diskusijų (2017–2018); kitų platformų žargonas, emoji, naujas slengas
  gali būti neatpažinti.
- Sarkazmas, ironija ir kontekstas už komentaro ribų (ankstesni pranešimai) nematomi.
- Retos kategorijos (`threat`, `identity_hate` < 1 %) atpažįstamos daug prasčiau.
- DistilBERT mato tik pirmus 128 žetonus; labai ilguose komentaruose agresija pabaigoje gali būti praleista.
- Anotacijos subjektyvios: žmonės ne visada sutaria, kas yra „toksiška“.
- DI žodžių filtras gali praleisti retus įžeidimus, kurių modelis beveik nematė, ir lėtesnis
  už paprastą tikrinimą (kelis kartus perskaičiuoja komentarą), todėl CSV režime valoma iki 300 komentarų.

## 9. Literatūra

1. Jigsaw / Conversation AI. *Toxic Comment Classification Challenge*, Kaggle, 2018.
   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
2. Sanh, V., Debut, L., Chaumond, J., Wolf, T. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* arXiv:1910.01108, 2019.
3. Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL, 2019.
4. Ribeiro, M. T., Singh, S., Guestrin, C. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME). KDD, 2016.
5. Dixon, L., Li, J., Sorensen, J., Thain, N., Vasserman, L. *Measuring and Mitigating Unintended Bias in Text Classification.* AIES, 2018.
6. Wulczyn, E., Thain, N., Dixon, L. *Ex Machina: Personal Attacks Seen at Scale.* WWW, 2017.
7. Pedregosa, F. ir kt. *Scikit-learn: Machine Learning in Python.* JMLR 12, 2011.
8. Wolf, T. ir kt. *Transformers: State-of-the-Art Natural Language Processing.* EMNLP (System Demos), 2020.
9. Idėjos (char n-gramos, žodžių santykinio dažnio analizė) iš viešo Kaggle sprendimo
   „Jigsaw Toxic Comment, 34th place“ ir jo aprašymo:
   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52645

## 10. Autoriai

| Narys | Atsakomybė |
|---|---|
| | Duomenys ir EDA |
| | Baseline modelis |
| | DistilBERT modelis |
| | Vertinimas, XAI, sąsaja, ataskaita |
