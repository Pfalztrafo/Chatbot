# Support-Chatbot zur Transformator-Kaufberatung

## Inhaltsverzeichnis
- [Projektübersicht](#projektübersicht)
- [Features](#features)
- [Verwendete Technologien](#verwendete-technologien)
- [Installation](#installation)
- [Training](#training)
- [Evaluierung](#evaluierung)
- [Verwendung](#verwendung)
- [Quellen und Lizenz](#quellen-und-lizenz)

---

## Projektübersicht

Dieses Projekt konzentriert sich auf die wissenschaftliche Analyse und Feinabstimmung von Large Language Models (LLMs) der FLAN-T5-Familie zur Entwicklung eines Support-Chatbots für die Transformator-Kaufberatung. Das Ziel ist es, branchenspezifische Anforderungen präzise zu bedienen und allgemeine Erkenntnisse zur Anpassung von LLMs für spezialisierte Anwendungen zu gewinnen.

### Hauptziele:
1. **Zwei-Phasen-Training:**
   - Phase 1: Nutzung von gefilterten GermanQuAD (352 Datensätze) und GermanDPR (655 Datensätze) zur Verbesserung der allgemeinen QA-Fähigkeiten.
   - Phase 2: Feinabstimmung mit domänenspezifischen FAQs (Sales und General).
2. **Evaluierung:**
   - Verwendung von BLEU und ROUGE zur Bewertung der Sprachqualität.
   - Qualitative Tests zur thematischen Präzision.
3. **Ergebnisse:**
   - Vergleich von FLAN-T5-Modellen (Small bis XL).
   - Optimierung von Modellparametern (z. B. Lernrate, Batchgröße).

---

## Features

- **Zwei-Phasen-Feinabstimmung:**
  - Kombination von gefilterten GermanQuAD, GermanDPR und domänenspezifischen FAQs.
- **Interne Prompts:** Optimierte Fragestellungen für präzisere Antworten.
- **Generative Modelle:** Einsatz von FLAN-T5-Small bis XL für präzise Sprachgenerierung.
- **GUI-Integration:** Steuerung von API, Training und Logs über eine benutzerfreundliche Oberfläche.
- **API-Schnittstelle:** Bereitstellung von Chatbot-Funktionen für externe Anwendungen.
- **Live-Chat:** Direkte Kommunikation mit dem Chatbot.
- **Einstellungen:** Anpassung von Parametern wie Temperatur und Modellwahl über die GUI.

---

## Verwendete Technologien

- **Modelle:** FLAN-T5 (Small, Base, Large, XL).
- **Datenquellen:**
  - Gefiltertes GermanQuAD: 352 Frage-Antwort-Paare aus Wikipedia.
  - Gefiltertes GermanDPR: 655 Frage-Antwort-Paare mit positiven und negativen Beispielen.
  - Domänenspezifische FAQs (Sales, General).
- **Bibliotheken:**
  - `fastapi`
  - `uvicorn`
  - `transformers`
  - `sentence_transformers`
  - `torch`
  - `nltk`
  - `psutil`
  - `requests`
  - `rouge_score`
  - `fuzzywuzzy`
  - `datasets`
  - `pydantic`
- **Entwicklungsumgebung:** Python 3.12, FastAPI, Tkinter.
- **Hardware:**
  - GPU-Server mit NVIDIA A100 GPUs.
  - Entwicklung auf lokalen Maschinen mit RTX 3050 (6 GB).

---

## Installation

### Abhängigkeiten installieren:

Führe den folgenden Befehl aus, um alle benötigten Bibliotheken zu installieren:
```bash
pip install fastapi uvicorn transformers sentence-transformers torch nltk psutil requests rouge-score fuzzywuzzy datasets pydantic
```

### Konfigurationsdatei:
- Passe die `config.json` an, um Modelle, Trainingseinstellungen und API-Parameter zu definieren.

---

## Training

### Zwei-Phasen-Training
1. **Phase 1:**
   - Nutzung von gefiltertem GermanQuAD und GermanDPR.
   - Ziel: Verbesserung der allgemeinen QA-Fähigkeiten.
2. **Phase 2:**
   - Feinabstimmung mit den domänenspezifischen FAQs (Sales, General).
   - Ziel: Spezifische Antworten auf Transformator-Fragen.

### Trainingsparameter
- Lernrate: 2e-5
- Batchgröße: 1-16 (abhängig von der GPU)
- Epochen: 3-5
- Negative-Beispiele-Rate: 50%

### Training starten:
```bash
python train_model.py
```

---

## Evaluierung

- **Quantitative Metriken:**
  - BLEU: Misst die Genauigkeit der generierten Antworten.
  - ROUGE: Bewertet die semantische Übereinstimmung.
- **Qualitative Tests:**
  - Präzision bei Transformator-spezifischen Fragen.
  - Umgang mit themenfremden Anfragen.

---

## Verwendung

### Mindestanforderungen
- Prozessor: Intel i5 oder besser
- RAM: 8 GB
- GPU: Optional (empfohlen für größere Modelle)

### Chatbot starten:
```bash
python chatbot_gui.py
```

### Funktionen:
1. **Live-Chat:** Echtzeitantworten auf Benutzeranfragen.
2. **API:** Integration in externe Anwendungen.
3. **GUI:** Verwaltung von Modellen, Training und Logs.
4. **Einstellungen:** Anpassung von Parametern wie Temperatur, Modellwahl und mehr.

---

## Quellen und Lizenz

### Quellen:
- [GermanQuAD](https://www.deepset.ai/germanquad)
- [GermanDPR](https://www.deepset.ai/germanquad)

### Lizenz:
Dieses Projekt unterliegt der MIT-Lizenz. Alle verwendeten Modelle und Datensätze unterliegen den jeweiligen Lizenzen.
