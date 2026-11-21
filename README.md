# 🛡️ Interneto komentarų agresijos filtras

**Tema Nr. 49.** Interneto komentarų agresijos filtravimas: prieš paskelbiant viešą komentarą
sistema perspėja vartotoją, jei jo žinutė turi toksiškumo požymių (patyčios, šauksmai).

Kai vartotojas spaudžia **„Skelbti“**, DI patikrina angliškai parašytą komentarą:
- **DI modelis** (TF-IDF + LogReg arba DistilBERT) įvertina agresiją: patyčias, įžeidimus, grasinimus, nešvankybes;
- **šauksmų detektorius** aptinka RAŠYMĄ DIDŽIOSIOMIS, „!!!“, ištęstus žodžius („STOOOP“).

Jei randama toksiškumo požymių, autorius **perspėjamas** ir jam **pasiūloma pataisyta versija**
(paslėpti įžeidžiantys žodžiai arba ta pati mintis be šauksmų). Galutinį sprendimą –
redaguoti, priimti pasiūlymą ar skelbti vis tiek – visada priima pats autorius.
Papildomai sąsajoje yra moderatoriaus režimas (paketinis CSV tikrinimas).

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
| Sprendimo kritiškumas | **Informacinis** (pataria) | Sistema nieko neblokuoja ir nešalina: tik perspėja autorių ir siūlo pataisymą, sprendžia žmogus |

## 2. Sistemos architektūra

```
  ✍️ Vartotojas rašo komentarą ir spaudžia „Skelbti“
                        │
         ┌──────────────┴──────────────────────────────┐
         ▼                                             ▼
 ┌──────────────┐                             ┌─────────────────┐
 │  Valymas ir  │ URL/IP/el. paštas → [URL]...│ Šauksmų detekto-│ DIDŽIOSIOS raidės,
 │anonimizavimas│                             │ rius (taisyklės)│ „!!!“, „STOOOP“
 └──────┬───────┘                             └────────┬────────┘
        ▼                                              │
 ┌──────────────┐   ┌──────────────┐                   │
 │ TF-IDF + LR  │ / │  DistilBERT  │                   │
 │  (Baseline)  │   │  (Proposed)  │                   │
 └──────┬───────┘   └──────┬───────┘                   │
        └────────┬─────────┘                           │
                 ▼                                     │
  7 tikimybės: agresyvus + 6 kategorijos               │
                 └──────────────────┬──────────────────┘
                                    ▼
          ┌─────────────────────────┼──────────────────────────┐
          ▼                         ▼                          ▼
  < 0.40 ir nešaukia      0.40–0.80 arba šauksmas     ≥ 0.80 (arba agresija + šauksmas)
  ✅ skelbiama iš karto   ⚠️ švelnus perspėjimas      ⛔ aiškus perspėjimas
                                    │
                                    ▼
     Pasiūlymas: 🧹 paslėpti įžeidžiantys žodžiai (DI žodžių filtras, XAI)
                 arba 🔉 ta pati mintis be šauksmų
                                    ▼
     👤 Autorius renkasi: priimti pasiūlymą / redaguoti / skelbti vis tiek
```

## 3. Failų struktūra

```
agresijos-filtras/
├── app.py                   # Streamlit sąsaja: perspėjimas autoriui + moderatoriaus režimas
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
│   ├── shouting.py          # šauksmų detektorius (DIDŽIOSIOS raidės, „!!!“, ištęsti žodžiai)
│   ├── warn.py              # perspėjimas prieš paskelbiant: DI modelis + šauksmai + pasiūlymas
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

Perspėjimas prieš paskelbiant ir šauksmų detektorius komandinėje eilutėje:

```bash
python -m src.warn "YOU ARE AN IDIOT!!!" "Thanks for the help"
python -m src.shouting "WHY DO YOU NEVER LISTEN!!! STOOOP"
# -> strong: DIDŽIOSIOS raidės (100 %), „!!!“, ištęsti žodžiai; ramiau: „Why do you never listen! Stop“
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
| **DistilBERT** | 258 | **69,0 %** | **1,8** | 0,90 → **0,32** |

*(300 atsitiktinių agresyvių test komentarų, 3–60 žodžių; `python -m src.censor --model distilbert --eval 300`,
Google Colab T4.)* Pvz. su DistilBERT: „This article is crap, whoever wrote it is a moron.“ – 99 % → 0,1 % paslėpus 2 žodžius;
„You are an idiot and nobody wants you here.“ – paslėpus „idiot“ lieka 48 %, nes frazė „nobody wants you here“
pati yra priešiška, todėl sprendžia moderatorius.
Baseline nepavykusių atvejų dažna priežastis – jis frazę „you are …“ laiko agresyvia net
be įžeidimų (pvz. „You are a nice person“ → 0,97), todėl tokie komentarai paliekami moderatoriui.

**Šauksmų detektorius (`src/shouting.py`).** Abu DI modeliai tekstą paverčia mažosiomis
raidėmis (TF-IDF `lowercase=True`, DistilBERT *uncased*), todėl šauksmų jie nemato. Šauksmas
pats savaime nebūtinai įžeidžia („THANK YOU SO MUCH!!!“), todėl jis vertinamas atskiru,
lengvai paaiškinamu taisyklių sluoksniu:

| Požymis | Slenkstis |
|---|---|
| Didžiųjų raidžių dalis (be santrumpų USA, NATO, BBC…; kai raidžių ≥ 8) | ≥ 70 % – stiprus, ≥ 40 % – švelnus |
| Žodžiai vien DIDŽIOSIOMIS (≥ 3 raidžių) | ≥ 2 žodžiai – švelnus |
| Šauktukų / klaustukų serijos | „!!!“ arba iš viso ≥ 4 „!“ |
| Ištęsti žodžiai („nooooo“, „STOOOP“) | tik kartu su kitais požymiais |

Lygis: 1 požymis – švelnus, ≥ 2 – stiprus šauksmas. Sistema pasiūlo ramesnę versiją:
*„WHY DOES NOBODY READ THE RULES??? THIS IS RIDICULOUS!!!“ → „Why does nobody read the rules? This is ridiculous!“*
Detektoriaus naudingumas patikrinamas EDA metu: kiek dažniau agresyvūs komentarai „rėkia“
(`reports/figures/eda_shouting.png`, `reports/metrics/eda_style.csv`).

**Perspėjimas prieš paskelbiant (`src/warn.py`, sąsajos skirtukas „✍️ Rašyti komentarą“).**

| Situacija | Perspėjimas | Pasiūlymas |
|---|---|---|
| Agresyvumas < 0,40, nešaukia | – (skelbiama iš karto) | – |
| Tik šauksmas | 📢 „Atrodo, kad jūs šaukiate“ | ta pati mintis be šauksmų |
| Agresyvumas 0,40–0,80 | ⚠️ „Žinutė gali būti suprasta kaip agresyvi“ | paslėpti įžeidžiantys žodžiai |
| Agresyvumas ≥ 0,80 arba agresija + stiprus šauksmas | ⛔ „Jūsų žinutė gali įžeisti kitus“ | paslėpti žodžiai; jei agresija slypi prasmėje – siūloma perrašyti |

Perspėjime nurodoma, *kodėl* (agresyvumo tikimybė, aptiktos kategorijos – įžeidimas, grasinimas,
neapykanta grupei – ir šauksmo požymiai). Autorius renkasi: **priimti pasiūlymą**, **redaguoti pats**
arba **skelbti vis tiek** (tuomet stipriai agresyvus komentaras pažymimas moderatoriui).
Sąsaja skaičiuoja sesijos statistiką: kiek komentarų patikrinta, kiek perspėta, kiek autorių pataisė.

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
- **Žmogus sprendimo cikle:** sistema nieko neblokuoja – ji tik perspėja autorių ir pasiūlo
  pataisymą, o sprendžia pats autorius (gali skelbti ir nepaisydamas perspėjimo). Moderatoriaus
  režime taip pat tik siūloma. Taip mažinama cenzūros rizika ir gerbiama žodžio laisvė.
- **Šauksmai ≠ agresija:** didžiosios raidės gali reikšti ir džiaugsmą („THANK YOU!!!“) ar būti
  regos sutrikimų turinčių žmonių įprotis, todėl vien šauksmas sukelia tik švelnų perspėjimą.
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
- Šauksmų detektorius paremtas taisyklėmis: santrumpų sąrašas ribotas (nauja santrumpa gali būti
  palaikyta šauksmu), o džiaugsmingas šauksmas („SO HAPPY!!!“) irgi sukelia švelnų perspėjimą.
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
| Neilas Rimkus | |
| Rokas Razokas | |
| Rokas Rapalavičius | |
