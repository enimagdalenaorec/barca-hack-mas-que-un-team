# Leave-one-game-out na 10 vanjskih utakmica (solution/logo.py)

Sve brojke u ovom dokumentu dolaze isključivo s vanjskih utakmica (3 Metrica sample + 7 IDSSE) ili
iz dokumentacije organizatora (DATA.md: lopta nedostaje na 41% ocjenjivanih frameova; u prosjeku
je praćeno 20.5 igrača). Natjecateljska utakmica nije korištena ni za što.

## Pravilo usvajanja (zapisano PRIJE mjerenja)

Za svaku held-out utakmicu model se trenira bez nje i boduje službenom bibliotekom
(`evaluation/possession_metrics.py`, `pm.evaluate`): spellF1 (`spell_f1_mean`) je glavna metrika,
macroF1 tie-break.

- **usvoji**: prosjek spellF1 na 10 utakmica raste za ≥ 0.01 prema referenci I varijanta
  pobjeđuje referencu na ≥ 7 od 10 utakmica;
- **možda**: prosjek raste, ali nije ispunjen barem jedan od ta dva uvjeta;
- **odbaci**: prosjek ne raste.

Ako se hiperparametar bira na held-out rezultatima (cijena prijelaza, STOP/RESTART, minimalna
duljina posjeda, CONTROL_S...), uz najbolju vrijednost prijavljuje se i **ugniježđena procjena**:
za svaku utakmicu vrijednost se bira na preostalih 9, primjenjuje na tu jednu, pa se uzima
prosjek. Odluka se donosi na ugniježđenoj procjeni, ne na optimističnom maksimumu.

Referenca: zadana konfiguracija predaje (trening 3 Metrica + 4 IDSSE J03WMX, J03WN1, J03WOH,
J03WOY bez held-out utakmice; SWITCH_COST 14; pravilo mrtve lopte MIN_DEAD_S 2 s + pre-kickoff
maska; zadana augmentacija; held-out degradiran istom zadanom augmentacijom s drugim seedom).
Kod eksperimenata koji mijenjaju degradaciju held-out utakmice (b, c) usporedba je uvijek s
referentnom konfiguracijom bodovanom pod ISTOM degradacijom.

## Kako se pokreće

```
.venv/bin/python -m solution.logo ref | idsse | dead | deadlag | noise | control | switch | ensemble | kickoff | combo | all
```

Harness (`solution/logo.py`) čita samo `data/external/cache/*.npz`, predikcije kešira u
`data/external/cache/logo/`, deterministički je (seed augmentacije = `seed_of(ime)`, held-out
`seed_of(ime) + 1`, kao `model.evaluate`). U tablicama: spellF1 = prosjek `spell_f1_mean` na 10
utakmica, zatim posebno 3 Metrica i 7 IDSSE; „pobjede" = na koliko od 10 utakmica varijanta ima
veći spellF1 od reference; „posjeda" = prosječan broj predviđenih / stvarnih posjeda po utakmici.

## Referenca

| utakmica | spellF1 |
|---|---|
| metrica1 | 0.754 |
| metrica2 | 0.778 |
| metrica3 | 0.701 |
| idsse_J03WMX | 0.543 |
| idsse_J03WN1 | 0.574 |
| idsse_J03WOH | 0.612 |
| idsse_J03WOY | 0.569 |
| idsse_J03WPY | 0.624 |
| idsse_J03WQQ | 0.658 |
| idsse_J03WR9 | 0.634 |
| **prosjek 10** | **0.645** (Metrica 0.744, IDSSE 0.602; macroF1 0.903; posjeda 308 / 412) |

Model predviđa oko četvrtinu manje posjeda nego što ih ima u oznakama; na IDSSE je rezultat
osjetno niži nego na Metrici (IDSSE oznake su sitnije: više kratkih posjeda).

## a. IDSSE 4 naspram 7

| varijanta | spellF1 (10) | Metrica | IDSSE | macroF1 | posjeda | Δ | pobjede | odluka |
|---|---|---|---|---|---|---|---|---|
| 3 Metrica + 4 IDSSE (ref) | 0.645 | 0.744 | 0.602 | 0.903 | 308 / 412 | referenca | | |
| 3 Metrica + 7 IDSSE | 0.641 | 0.733 | 0.602 | 0.904 | 299 / 412 | −0.004 | 5/10 | **odbaci** |

Tri dodatne IDSSE utakmice ne pomažu IDSSE held-out utakmicama (0.602 → 0.602), a Metrici
odmažu (dvije od tri padaju za 0.016–0.021). Napomena o usporedivosti: kad je held-out jedna od
4 referentne IDSSE utakmice, referentna grana trenira na 6, a šira na 9 utakmica; za tri IDSSE
utakmice izvan referentnog skupa 7 naspram 9. Ni u jednoj podskupini nema dosljedne dobiti.
`IDSSE_TRAIN` ostaje na 4.

## b. Mrtva lopta: pravilo naspram naučenog P(dead)

**Što se iz vanjskih podataka može izmjeriti.** Metrica (nakon premošćivanja rupa < 1 s), po
utakmici: bez lopte je 86% / 97% / 99% mrtvih frameova i 4% / 5% / 2% živih; ukupno 38% / 40% /
32%. Od 287 prekida lopta u 3.8% nikad ne nestane. Kad nestane, nestane praktički na istom
frameu na kojem prekid počinje i vrati se na frameu nastavka (90. percentil kašnjenja 0.1 s
odnosno 0.0 s); u 11% prekida rupa počinje prije prekida, u 7% završava nakon nastavka. Rupa
koje su cijele u živoj igri ima samo 30 na tri utakmice (duljine 25/50/75/90 pct: 1.5 / 2.0 /
4.0 / 21.6 s).

**Važno ograničenje.** Metricine rupe u lopti poklapaju se s prekidima na frame točno. To znači
da Metricin „nestanak lopte" nije ponašanje automatskog trackera, nego gotovo zapis statusa
lopte. Iz vanjskih podataka zato NIJE moguće izmjeriti koliko rupe pravog trackera kasne za
prekidom ili prethode nastavku. „Realistični" regim izveden iz Metrice je stoga po mrtvim
frameovima gotovo jednak zadanom regimu; razlikuje se po tome što (i) Metrica zadržava vlastite
prirodne rupe, (ii) na IDSSE svaki prekid posuđuje rupu slučajnog Metricina prekida slične
duljine (uklj. 3.8% prekida bez nestanka), (iii) rupe u živoj igri (Metricine duljine) dodaju se
dok ukupno 41% ocjenjivanih frameova nema loptu (DATA.md), osim gdje samo skrivanje prekida već
daje više.

| regim (held-out) | bez lopte: mrtvi | živi | ukupno | praćeno igrača |
|---|---|---|---|---|
| default, Metrica | 1.000 | 0.048 | 0.395 | 21.3 |
| default, IDSSE | 0.953 | 0.044 | 0.452 | 21.2 |
| harsh, Metrica | 0.998 | 0.119 | 0.440 | 21.3 |
| harsh, IDSSE | 0.867 | 0.116 | 0.452 | 21.2 |
| real, Metrica | 0.940 | 0.108 | 0.412 | 21.3 |
| real, IDSSE | 0.955 | 0.021 | 0.440 | 21.2 |

(IDSSE utakmice imaju 41–52% mrtvih frameova, pa već skrivanje prekida daje više od 41%.)

Naučeni model: `dead_model` + `viterbi3`, trening na istim utakmicama kao model momčadi (bez
held-out), SWITCH_COST 14. STOP/RESTART mreža {4, 8, 12, 20}², s ugniježđenim izborom. Referenca
u svakoj tablici: pravilo 2 s pod ISTIM held-out regimom.

**Held-out regim `default`**

| varijanta | spellF1 (10) | Metrica | IDSSE | macroF1 | posjeda | Δ | pobjede | odluka |
|---|---|---|---|---|---|---|---|---|
| pravilo 1 s | 0.646 | 0.746 | 0.603 | 0.903 | 309 / 412 | +0.001 | 3/10 | možda |
| pravilo 2 s | 0.645 | 0.744 | 0.602 | 0.903 | 308 / 412 | referenca | | |
| pravilo 3 s | 0.644 | 0.747 | 0.600 | 0.903 | 306 / 412 | −0.001 | 2/10 | odbaci |
| naučeni, STOP 20 / RESTART 8 (zadano) | 0.655 | 0.739 | 0.619 | 0.916 | 308 / 412 | +0.010 | 7/10 | možda (Δ 0.0099, tik ispod praga) |
| naučeni, ugniježđeni izbor (10× bira 12 / 20) | 0.655 | 0.739 | 0.619 | 0.916 | 308 / 412 | +0.010 | 7/10 | usvoji (tik iznad praga) |

**Held-out regim `harsh`**

| varijanta | spellF1 (10) | Metrica | IDSSE | macroF1 | posjeda | Δ | pobjede | odluka |
|---|---|---|---|---|---|---|---|---|
| pravilo 1 s | 0.596 | 0.701 | 0.551 | 0.839 | 299 / 412 | +0.001 | 4/10 | možda |
| pravilo 2 s | 0.595 | 0.698 | 0.551 | 0.839 | 298 / 412 | referenca | | |
| pravilo 3 s | 0.594 | 0.700 | 0.549 | 0.838 | 297 / 412 | −0.001 | 1/10 | odbaci |
| naučeni, STOP 20 / RESTART 8 (zadano) | 0.616 | 0.700 | 0.581 | 0.881 | 307 / 412 | +0.021 | 9/10 | **usvoji** |
| naučeni, ugniježđeni izbor (10× bira 20 / 20) | 0.618 | 0.700 | 0.582 | 0.881 | 305 / 412 | +0.023 | 9/10 | **usvoji** |

**Held-out regim `real`**

| varijanta | spellF1 (10) | Metrica | IDSSE | macroF1 | posjeda | Δ | pobjede | odluka |
|---|---|---|---|---|---|---|---|---|
| pravilo 1 s | 0.627 | 0.667 | 0.610 | 0.893 | 309 / 412 | −0.005 | 4/10 | odbaci |
| pravilo 2 s | 0.631 | 0.684 | 0.609 | 0.894 | 304 / 412 | referenca | | |
| pravilo 3 s | 0.634 | 0.698 | 0.607 | 0.894 | 300 / 412 | +0.003 | 3/10 | možda |
| naučeni, STOP 20 / RESTART 8 (zadano) | 0.639 | 0.691 | 0.617 | 0.907 | 305 / 412 | +0.007 | 9/10 | možda |
| naučeni, ugniježđeni izbor (8× bira 20 / 8) | 0.638 | 0.691 | 0.616 | 0.907 | 305 / 412 | +0.007 | 9/10 | možda |
| trening `real`: pravilo 2 s | 0.623 | 0.685 | 0.596 | 0.893 | 301 / 412 | −0.009 | 3/10 | odbaci |
| trening `real`: naučeni, ugniježđeni | 0.633 | 0.698 | 0.605 | 0.912 | 302 / 412 | +0.001 | 5/10 | možda |
| trening `harsh`: pravilo 2 s | 0.628 | 0.690 | 0.601 | 0.893 | 300 / 412 | −0.004 | 4/10 | odbaci |
| trening `harsh`: naučeni, ugniježđeni | 0.633 | 0.699 | 0.605 | 0.910 | 305 / 412 | +0.002 | 5/10 | možda |

Zaključci:

- MIN_DEAD_S 1 / 2 / 3 s: razlike ±0.001–0.005 bez dosljednog smjera. Ostaje 2 s.
- Naučeni P(dead) je bolji od pravila u sva tri regima (+0.010, +0.023, +0.007; 7, 9 i 9 pobjeda
  od 10) i uvijek diže macroF1 (+0.013 do +0.042). Dobit raste što se rupe u lopti manje
  poklapaju s prekidima (`harsh`). Na Metrici je neutralan do blago negativan u `default`
  (−0.005), a dobit dolazi s IDSSE.
- STOP/RESTART: mreža je ravna (razlika između zadanog 20 / 8 i najboljeg ≤ 0.002). Ostaje 20 / 8.
- Treniranje pod `real` ili `harsh` regimom ne pomaže (−0.004 do −0.009 uz pravilo; s naučenim
  modelom ±0.001). `AUG_REGIME` ostaje `default`.
- Upozorenje: u sva tri regima rupe počinju točno na prekidu, pa pravilo ovdje radi u gotovo
  idealnim uvjetima. Stres-test b2 (niže) provjerava što biva kad to ne vrijedi.

### b2. Stres-test: rupe koje se NE poklapaju s prekidima (pretpostavka, nije mjerenje)

Zadana augmentacija, ali lopta ostaje vidljiva do 3 s nakon prekida i vraća se do 3 s prije
nastavka (jednoliko). Ovo nije izvedeno iz podataka; služi samo da pokaže koliko koja metoda
ovisi o tome da rupa počinje točno na prekidu. Pogađa samo IDSSE (Metricine prirodne rupe ostaju).

| varijanta | spellF1 (10) | Metrica | IDSSE | macroF1 | posjeda | Δ | pobjede | odluka |
|---|---|---|---|---|---|---|---|---|
| pravilo 1 s | 0.582 | 0.738 | 0.515 | 0.870 | 332 / 412 | +0.002 | 5/10 | možda |
| pravilo 2 s | 0.579 | 0.737 | 0.512 | 0.869 | 330 / 412 | referenca | | |
| pravilo 3 s | 0.580 | 0.741 | 0.511 | 0.869 | 328 / 412 | +0.000 | 4/10 | možda |
| naučeni, STOP 20 / RESTART 8 (zadano) | 0.623 | 0.732 | 0.577 | 0.902 | 322 / 412 | +0.044 | 7/10 | **usvoji** |
| naučeni, ugniježđeni izbor (10× bira 20 / 20) | 0.624 | 0.732 | 0.577 | 0.902 | 321 / 412 | +0.044 | 7/10 | **usvoji** |
| trening `lag`: pravilo 2 s | 0.581 | 0.733 | 0.516 | 0.869 | 331 / 412 | +0.002 | 5/10 | možda |
| trening `lag`: naučeni, STOP 20 / RESTART 8 | 0.652 | 0.721 | 0.622 | 0.922 | 310 / 412 | +0.072 | 7/10 | **usvoji** |
| trening `lag`: naučeni, ugniježđeni izbor | 0.654 | 0.723 | 0.624 | 0.923 | 308 / 412 | +0.075 | 7/10 | **usvoji** |

Kad se rupe ne poklapaju s prekidima, pravilo na IDSSE pada s 0.602 na 0.512, a naučeni model
samo na 0.577 (i na 0.622 ako je i treniran pod takvim regimom). Svih 7 pobjeda su IDSSE
utakmice; tri Metrica utakmice su 0.002–0.006 niže (0.010–0.019 uz trening `lag`).

### b3. Isplati li se trenirati pod `lag` regimom?

| held-out | varijanta | spellF1 (10) | Metrica | IDSSE | macroF1 | Δ prema pravilu | pobjede |
|---|---|---|---|---|---|---|---|
| default | naučeni, zadani trening | 0.655 | 0.739 | 0.619 | 0.916 | +0.010 | 7/10 |
| default | naučeni, trening `lag` | 0.644 | 0.724 | 0.610 | 0.919 | −0.001 | 5/10 |
| harsh | naučeni, zadani trening | 0.616 | 0.700 | 0.581 | 0.881 | +0.021 | 9/10 |
| harsh | naučeni, trening `lag` | 0.622 | 0.696 | 0.590 | 0.902 | +0.027 | 9/10 |
| real | naučeni, zadani trening | 0.639 | 0.691 | 0.617 | 0.907 | +0.007 | 9/10 |
| real | naučeni, trening `lag` | 0.631 | 0.689 | 0.607 | 0.910 | −0.000 | 5/10 |
| lag | naučeni, zadani trening | 0.623 | 0.732 | 0.577 | 0.902 | +0.044 | 7/10 |
| lag | naučeni, trening `lag` | 0.652 | 0.721 | 0.622 | 0.922 | +0.072 | 7/10 |

Trening pod `lag` regimom je osiguranje, ne poboljšanje: dobiva +0.03 ako rupe stvarno kasne,
gubi 0.008–0.011 ako se poklapaju s prekidima. Koji je slučaj istinit vanjski podaci ne mogu
reći. Odluka: **možda** — ne uvoditi kao zadano bez dodatnog razloga.

### Kombinacija s cijenom prijelaza

Uz naučeni P(dead) najbolja cijena prijelaza: `default` 14 (10/10 ugniježđenih izbora; 0.655),
`harsh` 10 (0.619 naspram 0.616 za 14), `real` 10 (0.644 naspram 0.639 za 14). Plato je 8–14, pa
SWITCH_COST 14 ostaje i uz naučeni model; 10 bi bio jednako obranjiv izbor.

## c. Robusnost na šum

Held-out: zadana augmentacija, zatim glatki (Gaussov filtar, σ 1 s) šum pozicije na svakom igraču
i lopti te dodatna ispadanja igrača do ciljanog prosjeka praćenih. `mid`: igrači σ 0.25 m, lopta
σ 0.5 m, 20.5 praćenih (DATA.md). `high`: 0.5 m / 1.0 m, 19.5 praćenih. Veličina šuma je
pretpostavka; iz dokumentacije je poznat samo broj praćenih igrača.

| held-out | trening | spellF1 (10) | Metrica | IDSSE | macroF1 | Δ prema zadanom treningu | pobjede | odluka |
|---|---|---|---|---|---|---|---|---|
| čist | zadani | 0.645 | 0.744 | 0.602 | 0.903 | | | |
| čist | šum `mid` | 0.624 | 0.717 | 0.584 | 0.901 | −0.021 | | |
| čist | šum `high` | 0.603 | 0.704 | 0.560 | 0.896 | −0.042 | | |
| šum `mid` | zadani | 0.606 | 0.698 | 0.566 | 0.894 | referenca | | |
| šum `mid` | šum `mid` | 0.604 | 0.697 | 0.564 | 0.896 | −0.002 | 6/10 | odbaci |
| šum `mid` | šum `high` | 0.587 | 0.676 | 0.550 | 0.893 | −0.018 | 0/10 | odbaci |
| šum `high` | zadani | 0.564 | 0.644 | 0.530 | 0.882 | referenca | | |
| šum `high` | šum `mid` | 0.572 | 0.650 | 0.538 | 0.886 | +0.008 | 8/10 | možda |
| šum `high` | šum `high` | 0.571 | 0.660 | 0.532 | 0.887 | +0.007 | 9/10 | možda |

Šum košta: −0.039 (`mid`) i −0.081 (`high`) prema čistom held-outu. Treniranje sa šumom to ne
vraća: na umjerenom šumu ne pomaže, na jakom vraća manje od desetine pada, a na čistim podacima
košta 0.02–0.04. Ne uvoditi šum u augmentaciju.

## d. Kontrolirani posjed i minimalna duljina posjeda

Relabeliranje samo trening oznaka (`model.controlled`); held-out istina su izvorne oznake.

| varijanta | spellF1 (10) | Metrica | IDSSE | macroF1 | posjeda | Δ | pobjede | odluka |
|---|---|---|---|---|---|---|---|---|
| izvorne oznake (ref) | 0.645 | 0.744 | 0.602 | 0.903 | 308 / 412 | referenca | | |
| trening CONTROL_S 0.5 s | 0.640 | 0.739 | 0.598 | 0.903 | 307 / 412 | −0.004 | 1/10 | odbaci |
| trening CONTROL_S 1 s | 0.633 | 0.735 | 0.590 | 0.902 | 297 / 412 | −0.011 | 1/10 | odbaci |
| trening CONTROL_S 2 s | 0.597 | 0.706 | 0.551 | 0.898 | 271 / 412 | −0.047 | 0/10 | odbaci |
| CONTROL_S, ugniježđeni izbor (10× bira „bez") | 0.645 | | | | | 0.000 | 0/10 | odbaci |
| min. duljina 8 frameova | 0.645 | 0.744 | 0.602 | 0.903 | 308 / 412 | −0.000 | 0/10 | odbaci |
| min. duljina 12 | 0.644 | 0.744 | 0.601 | 0.903 | 307 / 412 | −0.001 | 1/10 | odbaci |
| min. duljina 25 | 0.641 | 0.740 | 0.598 | 0.903 | 297 / 412 | −0.004 | 1/10 | odbaci |
| min. duljina 38 | 0.621 | 0.721 | 0.578 | 0.901 | 275 / 412 | −0.024 | 0/10 | odbaci |
| min. duljina 50 | 0.598 | 0.709 | 0.550 | 0.898 | 255 / 412 | −0.047 | 0/10 | odbaci |
| min. duljina 75 | 0.550 | 0.653 | 0.506 | 0.892 | 224 / 412 | −0.095 | 0/10 | odbaci |
| min. duljina, ugniježđeni izbor (10× bira 0) | 0.645 | | | | | 0.000 | 0/10 | odbaci |

Oboje samo uklanja posjede, a model ih već predviđa premalo (308 naspram 412). Mjereno prema
izvornim oznakama izvora, svako daljnje stapanje kratkih posjeda gubi. Ograničenje: ovo mjeri
slaganje s granulacijom oznaka Metrice i IDSSE; ako su ciljne oznake grublje, zaključak ne mora
vrijediti, a to se iz vanjskih podataka ne može provjeriti. `CONTROL_S` ostaje `None`.

## e. Cijena prijelaza (SWITCH_COST)

| cijena | spellF1 (10) | Metrica | IDSSE | posjeda | Δ prema 14 | pobjede |
|---|---|---|---|---|---|---|
| 6 | 0.640 | 0.730 | 0.601 | 347 / 412 | −0.005 | 3/10 |
| 8 | 0.642 | 0.732 | 0.603 | 334 / 412 | −0.003 | 4/10 |
| 10 | 0.643 | 0.739 | 0.602 | 324 / 412 | −0.001 | 5/10 |
| 12 | 0.643 | 0.737 | 0.603 | 315 / 412 | −0.002 | 5/10 |
| **14** | **0.645** | **0.744** | 0.602 | 308 / 412 | referenca | |
| 17 | 0.640 | 0.741 | 0.597 | 298 / 412 | −0.005 | 0/10 |
| 20 | 0.636 | 0.736 | 0.594 | 290 / 412 | −0.008 | 1/10 |
| 25 | 0.629 | 0.730 | 0.585 | 279 / 412 | −0.016 | 0/10 |
| 30 | 0.619 | 0.723 | 0.575 | 271 / 412 | −0.026 | 0/10 |
| 40 | 0.601 | 0.712 | 0.553 | 256 / 412 | −0.044 | 0/10 |
| 50 | 0.590 | 0.701 | 0.543 | 246 / 412 | −0.054 | 0/10 |
| 60 | 0.575 | 0.689 | 0.527 | 236 / 412 | −0.069 | 0/10 |
| ugniježđeni izbor | 0.645 | 0.744 | 0.602 | | 0.000 | |

- Najbolja cijena na svih 10: **14**. Ugniježđeni izbor bira 14 u svih 10 preklopa, pa je
  ugniježđena procjena jednaka referenci (0.645): izbor nije optimističan.
- Po izvoru: Metrica 14 (0.744), IDSSE 8 (0.603), ali je IDSSE krivulja ravna od 6 do 14
  (0.601–0.603). Po utakmici: Metrica 14 / 20 / 14; IDSSE 14, 8, 6, 8, 8, 12, 8.
- **Novo, čisto obrazloženje za 14**: maksimum prosjeka spellF1 na svih 10 vanjskih held-out
  utakmica, izabran u 10 od 10 ugniježđenih preklopa; na IDSSE unutar 0.001 od najbolje
  vrijednosti, na Metrici najbolji. Plato je 8–14; iznad 17 rezultat pada na oba izvora.
  Javni prozor u izboru ne sudjeluje.

## f. Ansambl

HistGradientBoosting bez early stoppinga i s max_features = 1 ne ovisi o `random_state`, pa
članovi ansambla mijenjaju pomak subsamplea (koji od 5 isprepletenih skupova frameova) ili seed
augmentacije. Prosjek P(domaći) pa isti Viterbi.

| varijanta | spellF1 (10) | Metrica | IDSSE | macroF1 | posjeda | Δ | pobjede | odluka |
|---|---|---|---|---|---|---|---|---|
| jedan model (ref) | 0.645 | 0.744 | 0.602 | 0.903 | 308 / 412 | referenca | | |
| 3 pomaka subsamplea | 0.643 | 0.742 | 0.601 | 0.903 | 306 / 412 | −0.001 | 4/10 | odbaci |
| 5 pomaka subsamplea | 0.643 | 0.742 | 0.600 | 0.903 | 305 / 412 | −0.002 | 3/10 | odbaci |
| 3 seeda augmentacije | 0.641 | 0.743 | 0.598 | 0.903 | 299 / 412 | −0.003 | 3/10 | odbaci |
| 5 seedova augmentacije | 0.642 | 0.745 | 0.598 | 0.903 | 296 / 412 | −0.002 | 4/10 | odbaci |
| naučeni P(dead) + 5 pomaka za P(domaći) | 0.653 | 0.738 | 0.617 | 0.916 | 305 / 412 | +0.009 (naspram +0.010 bez ansambla) | 7/10 | odbaci |

Pojedinačni članovi (prosjek spellF1 na 10): pomaci 0–4: 0.645, 0.642, 0.643, 0.642, 0.641;
seedovi 0–4: 0.645, 0.634, 0.638, 0.641, 0.638. Dvije pouke:

- Ansambl je bolji od prosječnog člana (0.642 naspram 0.639 za seedove), ali ne od referentnog
  člana, koji je slučajno najbolji u obje obitelji. Prosjek zaglađuje P i dodatno smanjuje broj
  posjeda (296–306 naspram 308), što spellF1 ne nagrađuje.
- **Šum mjerenja**: sam seed augmentacije pomiče prosjek na 10 utakmica za do 0.011. Razlike
  manje od ~0.01 u ovom dokumentu zato treba čitati kao šum; prag iz pravila usvajanja je
  postavljen na pravu razinu. Isto vrijedi za referencu: njezinih 0.645 je na gornjem rubu
  raspona 0.634–0.645 koji daju drugi seedovi iste konfiguracije.

Nije pokrenuto (strojno vrijeme): 5 × (pomak + seed zajedno) i 5 × max_features 0.5 s različitim
`random_state` (definirani u `logo.ENSEMBLES_SKIPPED`).

## g. Početak poluvremena

Vanjske utakmice počinju živom igrom: u 17 od 20 poluvremena prvi je živi frame po oznakama
frame 0 poluvremena, u dva frame 1, u jednom frame 48. Vanjski podaci zato mogu izmjeriti samo
LAŽNE detekcije `kickoff_frame`, ne i to pronalazi li on stvarni početni udarac nakon duljeg
uvoda.

| postavka | točno | unutar 1 s | pogreške ≠ 0 (frameovi) |
|---|---|---|---|
| zadano (regim `default` i `real`) | 16/20 | 18/20 | +434, −1, −48, −1 |
| KICKOFF_SHARE 0.80 | 16/20 | 18/20 | +440, −1, −48, −1 |
| KICKOFF_SHARE 0.90 | 16/20 | 18/20 | +428, −1, −48, −1 |
| KICKOFF_HOLD_S 5 | 16/20 | 18/20 | +434, −1, −48, −1 |
| KICKOFF_HOLD_S 1.5 | 16/20 (`real`: 15/20) | 19/20 (18/20) | +434, −1, +6, −1 (`real`: i +40, +9) |
| KICKOFF_CALM_MS 2.0 | 16/20 | 18/20 | +434, −1, −48, −1 |

- Jedna lažna detekcija (oko 17 s živih frameova jedne Metrica utakmice proglašeno mrtvima) i
  jedan promašaj (48 frameova uvoda nije prepoznato). Nijedan od isprobanih parametara ne
  uklanja lažnu detekciju; HOLD 1.5 s hvata uvod od 48 frameova, ali u `real` regimu donosi dvije
  nove lažne detekcije.
- Utjecaj maske na spellF1: bez maske 0.645 naspram 0.645 s maskom (`default`), 0.632 naspram
  0.631 (`real`); razlika je < 0.001 i dolazi od one jedne lažne detekcije. Na vanjskim podacima
  maska ne može pomoći (nema uvoda), a šteta je zanemariva.
- Prvi posjed poluvremena: model + Viterbi pogađa momčad u 18/20 (`default`) i 19/20 (`real`);
  pravilo „momčad najbliža lopti u trenutku udarca" 18/20 i 18/20. Zamjena prvog posjeda tim
  pravilom ne mijenja spellF1 (0.000 / −0.000). **Odbaci**; ništa se ne mijenja.

## Sažetak i odluke

| eksperiment | najbolja varijanta | spellF1 (10) | Metrica / IDSSE | Δ | pobjede | odluka |
|---|---|---|---|---|---|---|
| referenca | | 0.645 | 0.744 / 0.602 | | | |
| a. IDSSE 4 → 7 | 7 IDSSE | 0.641 | 0.733 / 0.602 | −0.004 | 5/10 | odbaci |
| b. MIN_DEAD_S | 1 s | 0.646 | 0.746 / 0.603 | +0.001 | 3/10 | možda (šum) |
| b. naučeni P(dead), regim `default` | STOP 20 / RESTART 8 | 0.655 | 0.739 / 0.619 | +0.010 | 7/10 | na pragu (možda/usvoji) |
| b. naučeni P(dead), regim `harsh` | isto | 0.616 (ref 0.595) | 0.700 / 0.581 | +0.021 | 9/10 | usvoji |
| b. naučeni P(dead), regim `real` | isto | 0.639 (ref 0.631) | 0.691 / 0.617 | +0.007 | 9/10 | možda |
| b2. naučeni P(dead), stres-test `lag` | isto | 0.623 (ref 0.579) | 0.732 / 0.577 | +0.044 | 7/10 | usvoji |
| b. trening pod `real` / `harsh` | | 0.623–0.628 (ref 0.631) | | −0.004 do −0.009 | 3–4/10 | odbaci |
| b3. trening pod `lag` | | 0.644 (`default`) | 0.724 / 0.610 | −0.001 | 5/10 | možda (osiguranje) |
| c. trening sa šumom | šum `mid`, na šumnom held-outu | 0.604 (ref 0.606) | 0.697 / 0.564 | −0.002 | 6/10 | odbaci |
| d. CONTROL_S | 0.5 s | 0.640 | 0.739 / 0.598 | −0.004 | 1/10 | odbaci |
| d. min. duljina posjeda | 8 frameova | 0.645 | 0.744 / 0.602 | −0.000 | 0/10 | odbaci |
| e. SWITCH_COST | 14 (ugniježđeno: 14) | 0.645 | 0.744 / 0.602 | 0.000 | | ostaje 14, novo obrazloženje |
| f. ansambl | 5 pomaka subsamplea | 0.643 | 0.742 / 0.600 | −0.002 | 3/10 | odbaci |
| g. kickoff parametri, prvi posjed | | 0.645 | 0.744 / 0.602 | 0.000 | 0/10 | odbaci |

### Prijedlog

1. **`solution/pipeline.py`: `DEAD = "rule"` → `"model"`** (naučeni P(dead) + `viterbi3`, STOP 20 /
   RESTART 8, SWITCH_COST 14). Jedina promjena koja prolazi: bolja od pravila u sva četiri
   regima nestanka lopte (+0.007 do +0.044), 7–9 pobjeda od 10, macroF1 (tie-break) raste u svima
   (+0.013 do +0.042). U zadanom regimu dobit je točno na pragu (+0.010, 7/10) i reda je
   veličine šuma seeda; u regimima gdje se rupe manje poklapaju s prekidima jasno je iznad.
   Rizik: na 3 Metrica utakmice je 0.003–0.008 niži u zadanom regimu; dobit je u cijelosti s
   IDSSE. Kod već postoji; trud: jedna konstanta, ponovno pokretanje pipelinea i validacije
   (čovjek, ~15 min).
2. **`solution/decode.py`: SWITCH_COST ostaje 14**, ali komentar treba zamijeniti novim
   obrazloženjem (odjeljak e): maksimum na svih 10 vanjskih held-out utakmica, 10/10 ugniježđenih
   izbora. Trud: 5 min, bez promjene ponašanja.
3. Sve ostalo ostaje: `IDSSE_TRAIN` (4), `AUG_REGIME` (`default`), `CONTROL_S` (`None`),
   `MIN_DEAD_S` (2 s), `STOP_COST` / `RESTART_COST` (20 / 8), KICKOFF_* parametri, jedan model.

### Što se iz vanjskih podataka NE može zaključiti

- Koliko rupe u lopti automatskog trackera kasne za prekidima (Metricine rupe poklapaju se s
  prekidima na frame; IDSSE ima savršenu loptu). O tome ovisi veličina dobiti od naučenog P(dead).
- Radi li pre-kickoff maska na poluvremenu s duljim uvodom (vanjska poluvremena počinju živom
  igrom).
- Granulaciju ciljnih oznaka: odluke d i e mjere slaganje s oznakama Metrice i IDSSE.
- Stvarnu veličinu šuma pozicije (c je pretpostavka; dokumentiran je samo broj praćenih igrača).
- Kad je held-out Metrica utakmica, statistika rupa za `real` regim trening-utakmica uključuje
  i nju (samo duljine rupa, ne oznake); učinak je zanemariv, ali nije nula.

### Strojno vrijeme

Svi eksperimenti zajedno: oko 2 h 10 min na 10 jezgri (referenca + a: 6 min; e: 1 min; b: 25
min; c: 15 min; d: 9 min; g: 3 min; f: oko 45 min; kombinacija, b2 i b3: oko 25 min). Ponovno
pokretanje s punim kešom predikcija traje nekoliko minuta po eksperimentu.

# Krug 2 (solution/logo2.py)

Sve brojke i dalje dolaze isključivo s 10 vanjskih utakmica (3 Metrica + 7 IDSSE).

## Pravilo usvajanja (zapisano PRIJE mjerenja u ovom krugu)

Referenca je nova zadana konfiguracija predaje: team model (3 Metrica + 4 IDSSE bez held-out
utakmice) + naučeni P(dead) (isti trening skup) + `decode.viterbi3` sa SWITCH 14 / STOP 20 /
RESTART 8, zadana augmentacija treninga.

Mjerenje. U prvom krugu sam seed augmentacije held-out utakmice pomiče prosjek za do oko 0.011.
Zato se u ovom krugu svaka varijanta I referenca boduju na held-out utakmici degradiranoj s dva
seeda (`seed_of(ime) + 1` i `seed_of(ime) + 2`); prosjek spellF1 uzima se preko 20 parova
(utakmica × seed), a pobjede se broje po utakmici (prosjek spellF1 te utakmice preko 2 seeda).
Tri held-out regima iz `solution/logo.py`: `default`, `harsh` i `lag` (lopta vidljiva do 3 s nakon
prekida i prije nastavka; pretpostavka, stres-test).

- **usvoji**: na `default` prosjek spellF1 raste za ≥ 0.008 prema referenci I varijanta
  pobjeđuje na ≥ 7 od 10 utakmica I na `harsh` i na `lag` prosjek ne pada za više od 0.005;
- **možda**: prosjek na `default` raste, ali barem jedan uvjet nije ispunjen;
- **odbaci**: prosjek na `default` ne raste.

Ako se hiperparametar bira na held-out rezultatima, uz najbolju vrijednost prijavljuje se
**ugniježđena procjena** (za svaku utakmicu izbor na preostalih 9 po prosjeku `default`
spellF1, primjena na tu jednu, na oba seeda), i odluka se donosi na njoj, ne na optimističnom
maksimumu. Ista ugniježđena vrijednost po utakmici primjenjuje se i na `harsh` i `lag`.
Preferiraju se promjene koje ni u jednom regimu ne štete.

## Kako se pokreće (krug 2)

```
.venv/bin/python -m solution.logo2 prep                 # modeli po foldu, keš vjerojatnosti (jednom)
.venv/bin/python -m solution.logo2 ref | costs | evidence | flip | final | temp | dead7 | stack | deadgbm | smalldead | mpm
```

`prep` za svaki fold trenira jedan team model i dva dead modela (4 i 7 IDSSE) i sprema P(home),
P(dead), pre-kickoff masku i nekoliko stupaca dokaza za held-out utakmicu pod svakim regimom i
oba seeda u `data/external/cache/logo2/`. Svi eksperimenti dekodiranja rade na tom kešu. Bodovanje
je `pm.spell_scores` / `pm.evaluate` iz službene biblioteke. `final` čita mrežu koju spremi `flip`.

## Referenca (2 seeda)

| regim | spellF1 (20) | Metrica | IDSSE | macroF1 | posjeda pred./istina | seed +1 / seed +2 |
|---|---|---|---|---|---|---|
| default | 0.6526 | 0.740 | 0.615 | 0.915 | 308 / 412 | 0.6545 / 0.6506 |
| harsh | 0.6157 | 0.699 | 0.580 | 0.878 | 310 / 412 | 0.6164 / 0.6149 |
| lag | 0.6239 | 0.735 | 0.576 | 0.901 | 322 / 412 | 0.6234 / 0.6243 |

## Rezultati po ideji

Δ je prema referenci u istom regimu; „ugniježđeno" = izbor na 9, primjena na 10. utakmicu.

| # | ideja | najbolje na svih 10 (optimistično) | ugniježđeno, default | pobjede | Δ harsh / lag | odluka |
|---|---|---|---|---|---|---|
| 1 | SWITCH × STOP × RESTART, 120 postavki | (12, 30, 20): +0.0010 | +0.0005 | 7/10 | +0.0027 / −0.0003 | možda (šum); cijene ostaju |
| 2 | cijena prijelaza ovisna o dokazu, 108 postavki: skok P(home) | +0.0022 | | | | možda (šum) |
| 2 | isto: viša/niža cijena dok lopte nema | +0.0012 | | | | možda (šum) |
| 2 | isto: niža cijena oko promjene najbliže momčadi („flip") | +0.0070 | +0.0053 | 9/10 | +0.0045 / +0.0048 | možda → šira mreža (2b) |
| 3 | temperatura T_home × T_dead, 49 postavki | (0.7, 2.0): +0.0019 | +0.0005 | 5/10 | +0.0034 / −0.0000 | možda (šum) |
| 3 | samo T_dead 2 | +0.0009 | | 8/10 | +0.0020 / +0.0017 | možda |
| 4 | P(dead) treniran na 3 Metrica + 7 IDSSE | +0.0022 (macroF1 +0.006) | (nema izbora) | 9/10 | +0.0054 / +0.0153 | možda; nigdje ne šteti |
| 5 | drugi stupanj (team + dead), cijene birane ugniježđeno | (6, 20, 8): +0.0043 | +0.0043 | 6/10 | +0.0106 / +0.0204 | možda; ne isplati se |
| 5 | drugi stupanj, zadane cijene | −0.0139 | | 0/10 | −0.0089 / +0.0043 | odbaci |
| 7 | posteriorno dekodiranje (max. marginalna), 16 postavki | (14, 20, 8): +0.0003 | +0.0003 | 5/10 | −0.0000 / −0.0000 | odbaci (isto kao Viterbi) |

Napomene:

- **1.** Mreža je ravna (medijan 0.6514, maksimum 0.6535). SWITCH 14 uz naučeni P(dead) nije
  pogrešno biran; STOP i RESTART ne znače gotovo ništa.
- **2.** „Flip" je frame na kojem se momčad najbliža lopti promijeni (obje udaljenosti poznate na
  oba framea) i novi najbliži igrač je unutar R metara. U prozoru ±W frameova oko flipa cijena
  prijelaza je SWITCH × r, drugdje SWITCH. Kratki posjed koji počinje i završava na takvim
  frameovima više ne mora nadjačati dvije pune cijene; izvan njih cijena smije biti viša (20), pa
  nema treperenja.
- **5.** Drugi stupanj: GBM (200 stabala) nad P(home), P(dead) i njihovim kliznim prosjecima
  ±0.4/1/3/10 s, treniran na out-of-fold vjerojatnostima prvog stupnja (28 dodatnih parova
  modela). Zaglađuje vjerojatnosti pa sa zadanim cijenama gubi kratke posjede; s nižom cijenom
  prijelaza vraća se malo iznad reference. Dobit ne opravdava dodatni stupanj treninga.

## 2b. Flip na široj mreži, zajedno s P(dead) na 7 IDSSE i T_dead

Mreža: SWITCH {17, 20, 25, 30} × r {0.1, 0.2, 0.3, 0.4} × R {1, 1.5, 2 m} × W {6, 10, 15} × P(dead)
{4, 7 IDSSE} × T_dead {1, 2} = 576 postavki; ugniježđeni izbor ide preko cijele (pod)mreže.

| varijanta | spellF1 default (20) | Metrica | IDSSE | posjeda pred./istina | Δ | pobjede | Δ harsh | Δ lag | odluka |
|---|---|---|---|---|---|---|---|---|---|
| referenca | 0.6526 | 0.740 | 0.615 | 308 / 412 | referenca | | | | |
| samo P(dead) na 7 IDSSE | 0.6548 | 0.743 | 0.617 | 308 / 412 | +0.0022 | 9/10 | +0.0054 | +0.0153 | možda |
| samo T_dead 2 | 0.6535 | 0.741 | 0.616 | 307 / 412 | +0.0009 | 8/10 | +0.0020 | +0.0017 | možda |
| flip (20, 0.4, 2.0, 10) (fiksno) | 0.6605 | 0.747 | 0.624 | 310 / 412 | +0.0079 | 10/10 | +0.0046 | +0.0080 | možda |
| flip, ugniježđeni izbor | 0.6586 | 0.745 | 0.621 | 314 / 412 | +0.0060 | 10/10 | +0.0035 | +0.0065 | možda |
| flip + dead7 (fiksno) | 0.6629 | 0.749 | 0.626 | 309 / 412 | +0.0103 | 10/10 | +0.0096 | +0.0239 | usvoji |
| flip + dead7, ugniježđeni izbor | 0.6612 | 0.749 | 0.624 | 313 / 412 | +0.0086 | 10/10 | +0.0084 | +0.0223 | usvoji |
| **flip + dead7 + T_dead 2 (fiksno, prijedlog)** | **0.6641** | 0.751 | 0.627 | 307 / 412 | **+0.0116** | 10/10 | +0.0118 | +0.0261 | usvoji |
| flip (20, 0.3, 2.0, 10) + dead7 + T_dead 2 (fiksno, najbolji na svih 10) | 0.6643 | 0.751 | 0.627 | 314 / 412 | +0.0117 | 10/10 | +0.0118 | +0.0254 | usvoji |
| **flip + dead7 + T_dead 2, ugniježđeni izbor** | **0.6623** | 0.749 | 0.625 | 311 / 412 | **+0.0097** | 10/10 | +0.0107 | +0.0243 | **usvoji** |

Ugniježđeni izbori (puna mreža): (20, 0.3, 2.0, 10) u 5 foldova, (20, 0.3, 2.0, 6) u 3,
(20, 0.4, 2.0, 10) u 2; uvijek dead7 i T_dead 2. R = 2 m je rub mreže.

Po utakmici (prosjek 2 seeda), referenca → prijedlog: metrica1 0.746 → 0.756, metrica2 0.772 →
0.786, metrica3 0.703 → 0.711, J03WMX 0.571 → 0.582, J03WN1 0.570 → 0.579, J03WOH 0.648 → 0.665,
J03WOY 0.585 → 0.603, J03WPY 0.624 → 0.633, J03WQQ 0.666 → 0.668, J03WR9 0.640 → 0.658.

### Provjera izvan pravila: šum pozicije

Flip ovisi o točnim pozicijama lopte i igrača, a regimi `harsh` i `lag` mijenjaju samo rupe u
lopti. Zato je dodana provjera pod glatkim šumom pozicije iz prvog kruga (`mid`: igrači σ 0.25 m,
lopta σ 0.5 m; `high`: 0.5 m / 1.0 m i jedan praćeni igrač manje; pretpostavka, ne mjerenje),
s istim modelima, 2 seeda. Referenca: 0.6108 (`mid`), 0.5634 (`high`).

| varijanta | macroF1 default (Δ) | Δ šum mid (pobjede) | Δ šum high (pobjede) |
|---|---|---|---|
| samo P(dead) na 7 IDSSE | 0.9211 (+0.0059) | +0.0002 (6/10) | −0.0007 (5/10) |
| samo T_dead 2 | 0.9152 (+0.0000) | +0.0007 (7/10) | +0.0005 (8/10) |
| flip (20, 0.4, 2.0, 10) | 0.9160 (+0.0008) | +0.0012 (7/10) | −0.0009 (3/10) |
| flip + dead7 + T_dead 2 (prijedlog) | 0.9222 (+0.0070) | +0.0023 (6/10) | −0.0009 (4/10) |
| flip (20, 0.3, 2.0, 10) + dead7 + T_dead 2 | 0.9223 (+0.0071) | +0.0011 (5/10) | −0.0032 (4/10) |
| flip + dead7 + T_dead 2, ugniježđeni izbor | 0.9220 (+0.0068) | +0.0013 (6/10) | −0.0026 (4/10) |

**Dobit od flipa nestaje kad su pozicije šumne**: pod `mid` je unutar šuma mjerenja, pod `high`
blago negativna. Zato se predlaže r = 0.4 umjesto 0.3: na `default` je isti (−0.0001), a pod
šumom gubi manje (−0.0009 naspram −0.0032). Zaglađivanje udaljenosti (±2 do ±7 frameova) i
histereza (0.5 / 1 m) prije traženja flipova isprobani su i ne pomažu: smanjuju dobit na
`default`, a pod `high` ostaju oko nule.

## 6. GBM postavke dead modela

Jedna po jedna promjena prema zadanima (600 stabala, lr 0.1, 63 lista, min_samples_leaf 200).

| varijanta | spellF1 default (20) | Δ | pobjede | Δ harsh | Δ lag | odluka |
|---|---|---|---|---|---|---|
| 200 stabala | 0.6528 | +0.0002 | 7/10 | +0.0002 | +0.0012 | možda (šum) |
| lr 0.05 | 0.6526 | +0.0001 | 6/10 | −0.0005 | −0.0004 | možda (šum) |
| min_samples_leaf 50 | 0.6531 | +0.0006 | 7/10 | +0.0007 | −0.0003 | možda (šum) |
| min_samples_leaf 1000 | 0.6523 | −0.0003 | 5/10 | – | – | odbaci |
| 15 listova | 0.6532 | +0.0006 | 5/10 | +0.0032 | +0.0059 | možda |
| 100 stabala, 15 listova | 0.6564 | +0.0038 | 9/10 | +0.0081 | +0.0191 | možda |

Manji dead model (100 stabala, 15 listova) jedini se izdvaja, i to u istom smjeru kao T_dead 2 i
dead7 (manje samouvjeren P(dead)). Povrh prijedloga (`python -m solution.logo2 smalldead`) na
`default` ne dodaje ništa (+0.0000 na 4 IDSSE, +0.0007 na 7 IDSSE uz T_dead 2), a u stres-regimima
dodaje +0.005 (`harsh`) i +0.008 (`lag`). Ne ispunjava pravilo i nije u prijedlogu; za njega u
predaji nema koda (trebao bi vlastite GBM postavke dead modela).

## Odluka i prijedlog (krug 2)

Pravilo ispunjava samo kombinacija **flip + P(dead) na 7 IDSSE + T_dead 2**: ugniježđena procjena
+0.0097 na `default`, 10/10 utakmica, `harsh` +0.0107, `lag` +0.0243, macroF1 +0.007. Sam flip
(+0.0060 ugniježđeno) i sami dead7 / T_dead ostaju ispod praga 0.008.

Predložene promjene zadanih vrijednosti (kod je dodan i po zadanom isključen):

1. `solution/decode.py`: `SWITCH_COST` 14.0 → **20.0**
2. `solution/decode.py`: `FLIP_FACTOR` None → **0.4** (`FLIP_RADIUS_M` 2.0 i `FLIP_WINDOW` 10 već
   stoje)
3. `solution/decode.py`: `DEAD_TEMPERATURE` None → **2.0**
4. `solution/dead_model.py`: `IDSSE_TRAIN` "team" → **None** (dead model na svih 7 IDSSE; team
   model ostaje na `model.IDSSE_TRAIN`, 4 utakmice)

1 i 2 idu zajedno: SWITCH 20 bez flipa nije mjeren kao zasebna promjena i nije prijedlog.
Konzervativnija inačica je samo 4 (+0.002, 9/10, bez štete u ijednom regimu); ne ispunjava prag.

Rizik: dobit od flipa mjerena je na čistim pozicijama vanjskih utakmica. Pod pretpostavljenim
šumom pozicije svodi se na nulu (`mid`) ili na −0.001 (`high`); koliki je šum automatskog
praćenja, iz vanjskih podataka nije poznato. Dobit dead7 + T_dead ne ovisi o tome.

### Strojno vrijeme (krug 2)

Oko 1 h 30 min na 10 jezgri: `prep` 9 min (3 regima) + 24 min (dva regima šuma, uz druge poslove),
`stack` 33 min (uz druge poslove), `flip` 4 min, `deadgbm` 11 min, `smalldead` 3 min, ostalo oko
10 min.

# Krug 3: dodiri lopte (solution/touch_exp.py)

Ideja: posjed se mijenja u trenutku *dodira*, a sve dotadašnje značajke pitaju samo tko je
*blizu* lopte. Dodir = nagla promjena brzine lopte (≥ 5 m/s između 3 framea prije i 3 poslije,
lokalni vrh), pripisan momčadi čiji je najbliži igrač unutar 2.5 m i barem 0.5 m bliži od
protivnika. Kontrola = lopta sporija od 12 m/s unutar 1.5 m igrača jedne momčadi bez protivnika
jednako blizu. 14 novih značajki (`features.TOUCH_FEATURES`): tko je zadnji dotaknuo / tko će
sljedeći, sekunde od / do dodira i kontrole svake momčadi, broj dodira u 3 s prije / poslije.
Ideja prema hadisotudeh/more-than-a-hack-2027; kod je naš.

Isti harness i isto pravilo usvajanja kao krug 2: 10 vanjskih utakmica, 2 seeda, dekodiranje
točno kao u predaji (viterbi3 + flip + P(dead) na 7 IDSSE + T_dead 2). Referenca (bez dodira)
ponavlja predanu konfiguraciju: 0.6641.

| varijanta | default | Metrica | IDSSE | macroF1 | harsh | lag | posjeda pred./istina | pobjede | odluka |
|---|---|---|---|---|---|---|---|---|---|
| bez dodira (referenca) | 0.6641 | 0.751 | 0.627 | 0.922 | 0.6275 | 0.6499 | 307 / 412 | | |
| dodiri, lopta nezaglađena | 0.7527 | 0.834 | 0.718 | 0.941 | 0.7132 | 0.7368 | 339 / 412 | 10/10 | usvoji |
| **dodiri, lopta zaglađena 5 frameova** | **0.7477** | 0.845 | 0.706 | 0.939 | 0.7078 | 0.7343 | 334 / 412 | 10/10 | **usvoji** |

Sigurnosne provjere izvan pravila (Δ prema referenci, pobjede):

| regim | nezaglađeno | zaglađeno (5) |
|---|---|---|
| šum mid (glatki, logo.NOISE) | +0.0850 (10/10) | – |
| šum high | +0.0216 (8/10) | – |
| jitter lopte 10 cm po frameu | +0.0291 (9/10) | +0.0908 (10/10) |
| jitter lopte 20 cm po frameu | −0.0398 (2/10) | +0.1077 (10/10) |

Nezaglađeni detektor jitter od 20 cm pretvara u lažne dodire. Zaglađivanje (pomični prosjek
5 frameova prije detekcije) to rješava uz cijenu od 0.005 na default regimu; ne znamo koliko
natjecateljski tracker trese loptu, pa je usvojena zaglađena verzija. `python -m solution.touch_check`
(pokreće čovjek) uspoređuje broj dodira po minuti igre na natjecateljskoj utakmici s vanjskima
(~24–28) kao provjeru da tracker nije previše šumovit.

## Krug 3b: natjecateljski tracker zaglađuje loptu

`python -m solution.touch_check` (pokreće čovjek): oštri detektor nalazi 21.7 dodira u minuti igre
na vanjskim utakmicama, a **0.1** na natjecateljskoj. `python -m solution.touch_diag` (agregati,
pokreće čovjek): najveće promjene brzine lopte kroz 3 framea su na našoj utakmici petina vanjskih
(p99 3.8 naspram 16.8 m/s), a najbliži igrač je pri njima 4.6 m daleko naspram 0.5 m. Tracker
zaglađuje putanju lopte, pa oštri detektor na našoj utakmici ne vidi ništa; ta verzija **nije
predana**.

Popravak: lopta se prije detekcije zaglađuje 25 frameova (1 s) na svakoj utakmici, više nego
ijedan tracker, pa detektor više ne ovisi o trackeru. Pragovi postavljeni samo na vanjskim
utakmicama za tako zaglađenu loptu (pravi udarci postaju skokovi od 3–8 m/s kroz 8 frameova,
najbliži igrač medijan 1.9 m): TOUCH_SMOOTH 25, TOUCH_LAG 8, TOUCH_MIN_DV 3.0, TOUCH_RADIUS_M 3.5.
Provjera (ne podešavanje) na natjecateljskoj utakmici: 14.7 dodira u minuti naspram 15.9–22.5 na
vanjskima.

| regim | bez dodira | dodiri, zaglađeno 1 s | Δ | pobjede |
|---|---|---|---|---|
| default | 0.6641 | **0.7026** | +0.0385 | 8/10 |
| harsh | 0.6275 | 0.6657 | +0.0382 | 10/10 |
| lag | 0.6499 | 0.6874 | +0.0374 | 9/10 |
| jitter 20 cm | 0.6284 | 0.6826 | +0.0542 | 10/10 |

Metrica 0.790, IDSSE 0.665, macroF1 0.931, 321 / 412 posjeda. Odluka: **usvoji** (dobitak ≥ 0.008,
8/10 ≥ 7, harsh i lag pozitivni). Predana konfiguracija.
