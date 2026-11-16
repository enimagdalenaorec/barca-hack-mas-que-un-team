# Pravila za AI agente u ovom repozitoriju

Uvjeti natjecanja (FAQ.md, OVERVIEW.md) zabranjuju da videi, podaci utakmice ili BILO KAKVI
izvaci iz njih dođu do cloud LLM-a. Claude je cloud LLM. Zato:

- Nikad ne čitaj, ne listaj, ne ispisuj i ne pokreći ništa što učitava `data/tactical`,
  `data/broadcast`, `data/cache`, `*.mp4`, `evaluation/gt`, `evaluation/results`,
  `evaluation/reports`, `evaluation/scoreboard.csv` ili starter notebooke.
- Nikad ne pokreći `solution.pipeline`, `solution.tracking`, `solution.diagnostics`,
  `evaluation/run_evaluation.py`, `evaluation/validate_submission.py` ni starter loadere.
  To pokreće čovjek u svom terminalu. Nikad ne traži da ti zalijepi izlaz, ni agregatne brojke.
- U kodu, komentarima, dokumentima i commitovima ne smije biti brojki ni zapažanja o utakmici,
  osim onih iz dokumentacije organizatora.
- Sve učenje, podešavanje i odabir modela radi se samo na vanjskim utakmicama (`data/external/`).
  Javnih 5 minuta nikad ne odlučuje između verzija. Nema ručnih oznaka na utakmici.
- Ako dokument tima (npr. docs/experiments.md) sadrži brojke o utakmici ili javnom prozoru,
  preskoči ih i ne koristi.
- Svakom subagentu ova pravila prepiši u prompt.
- Ne pushaj bez izričitog odobrenja.
