# Pravila za AI agente u ovom repozitoriju

Uvjeti natjecanja (FAQ.md, OVERVIEW.md) zabranjuju da videi i podaci utakmice ili izvaci iz njih
dođu do cloud LLM-a. Claude je cloud LLM. Agregatni rezultati (spellF1, macroF1, broj posjeda,
brojke u dokumentima tima) nisu izvadak podataka i smiju se čitati. Zato:

- Nikad ne čitaj, ne listaj, ne ispisuj i ne pokreći ništa što učitava sirove podatke utakmice ili
  oznake po frameu: `data/tactical`, `data/broadcast`, `data/cache`, `*.mp4`, `evaluation/gt`,
  `evaluation/results` (CSV predaje po frameu) ili starter notebooke.
- Nikad ne pokreći `solution.pipeline`, `solution.tracking`, `solution.diagnostics`,
  `evaluation/run_evaluation.py`, `evaluation/validate_submission.py` ni starter loadere.
  To pokreće čovjek u svom terminalu.
- Agregatne rezultate smiješ čitati: dokumente tima (docs/experiments.md,
  docs/technical_document.md), sažetke rezultata i brojke koje ti čovjek zalijepi. Ne traži niti
  ispisuj oznake po frameu, koordinate ni druge sirove podatke.
- Sve učenje, podešavanje i odabir modela radi se samo na vanjskim utakmicama (`data/external/`).
  Javnih 5 minuta nikad ne odlučuje između verzija (EVALUATION.md); rezultat na njemu služi samo
  kao provjera da pipeline radi. Nema ručnih oznaka na utakmici. Konstante u kodu ne smiju biti
  podešene prema rezultatima ili zapažanjima s utakmice.
- Svakom subagentu ova pravila prepiši u prompt.
- Ne pushaj bez izričitog odobrenja.
