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
