# Support-Chatbot im Transformatorensegment

Dieser Chatbot wurde entwickelt, um Kunden bei Kaufempfehlungen und allgemeinen Fragen zu Transformatoren zu unterstützen. Er bietet Informationen zu Transformator-Typen, Services vom Unternehmen Pfalztrafo und stellt Empfehlungen basierend auf spezifischen Kundenanfragen zur Verfügung.

## Inhaltsverzeichnis
- [Übersicht](#übersicht)
- [Features](#features)
- [Verwendete Technologien](#verwendete-technologien)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Training](#training)
- [Versionshistorie](#versionshistorie)
- [Lizenz](#lizenz)
- [Quellen](#quellen)
- [Haftungsausschluss](#haftungsausschluss)


## Übersicht
Der Support-Chatbot im Transformatorensegment basiert auf einem feinabgestimmten Sprachmodell (fine-tuned LLM), das auf `google/flan-t5-base` basiert. Mithilfe von Retrieval-Augmented Generation (RAG) kann der Bot dynamisch auf Kundenfragen reagieren und gibt dabei gezielte Informationen zu Transformatoren und Dienstleistungen von Pfalztrafo.

## Features
- **FAQ-Antworten**: Beantwortet häufig gestellte Fragen zu Transformatoren und Pfalztrafo-Dienstleistungen.
- **Kaufempfehlungen**: Gibt Empfehlungen basierend auf spezifischen Anwendungen und Anforderungen der Kunden.
- **Synonym- und Fuzzy-Matching**: Erkennt ähnliche oder leicht abweichende Anfragen.
- **Retrieval-Augmented Generation (RAG)**: Kombiniert Wissensdatenbanken mit Modellantworten für präzise und aktuelle Antworten.
- **Mehrsprachige Unterstützung**: Basierend auf `google/flan-t5-base` unterstützt das Modell mehrere Sprachen.

## Verwendete Technologien
- **Visual Studio Code**: Entwicklungsumgebung für die Implementierung.
- **Python 3.12**: Programmiersprache für die Implementierung.
- **Hugging Face Transformers**: Für die Modellarchitektur und das Sprachverständnis.
- **LangChain**: Zum Aufbau und zur Verwaltung der Dialogstruktur und RAG-Ketten.
- **FAISS**: Vektorsuche für die Wissensdatenbank (läuft aktuell auf der CPU).
- **Fuzzywuzzy**: Für die Erkennung ähnlicher Anfragen (Fuzzy Matching).

## Projektstruktur
Die Projektstruktur ist in mehrere Module und Datenquellen unterteilt:

```plaintext
├── chatbot_main.py              # Hauptskript zum Ausführen des Chatbots
├── train_model.py               # Trainingsskript zum Feinabstimmen des Modells
├── utils.py                     # Konfiguration und Hilfsfunktionen
├── data/
│   ├── dialogues.json           # Häufig gestellte Fragen und Antworten
│   ├── trafo_info.json          # Allgemeine Infos über Transformatoren
│   ├── pfalztrafo_services.json # Dienstleistungen von Pfalztrafo
│   ├── rules.json               # Entscheidungsregeln für Empfehlungen
├── modules/
│   ├── dialogue_manager.py      # Modul zur Verwaltung von Dialogen
│   ├── knowledge_base.py        # Modul für den Zugriff auf Transformatoren-Wissen
│   ├── recommendation.py        # Modul für Kaufempfehlungen und Entscheidungslogik
├── fine_tuned_model/            # Verzeichnis für das trainierte Modell
├── .gitignore                   # Ausschließen der Git-Dateien wie fine_tuned_model wegen Speichergröße 
└── README.md                    # Diese Dokumentation
```

## Installation

**Abhängigkeiten installieren**:
```plaintext
pip install torch
pip install langchain
pip install langchain langchain-community
pip install langchain-huggingface
pip install -U langchain
pip install datasets
pip install transformers[torch]
pip install faiss-gpu
pip install faiss-cpu
pip install fuzzywuzzy[speedup]
```

## Verwendung

### Mindestanforderungen für die Nutzung
Um den Chatbot ohne Training zu nutzen, sind die folgenden minimalen Systemanforderungen empfohlen:

- **Prozessor**: Intel Core i3 oder AMD Ryzen 3 (mindestens Dual-Core)
- **RAM**: 8 GB
- **Speicherplatz**: Etwa 4 GB für das vortrainierte Modell und die Wissensdatenbanken
- **Grafikkarte**: Keine dedizierte GPU erforderlich; eine integrierte GPU reicht aus
- **Betriebssystem**: Windows 10 oder höher, macOS, oder eine Linux-Distribution mit Python 3.8+

Diese Konfiguration ermöglicht eine reibungslose Nutzung des Chatbots für einfache Anfragen in Echtzeit.

1. **Starten des Chatbots**:
   ```bash
   python chatbot_main.py
    ```
2. **Interaktive Eingabe**:
Geben Sie Fragen ein, wie z. B.:
"Welche Transformator-Typen gibt es?"
"Was kostet ein Transformator?"

3. **Beenden des Chats**:
Geben Sie `exit` ein, um die Chat-Sitzung zu beenden.


## Training
### Mindestanforderungen für das Training
Das Training des Modells erfordert deutlich höhere Systemressourcen als die Nutzung. Die folgenden Mindestanforderungen ermöglichen ein akzeptables Trainingstempo:
- **Prozessor**: Intel Core i5 oder AMD Ryzen 5 (Quad-Core oder höher)
- **RAM**: 16 GB (Mehr ist empfohlen, um Trainingsdaten effizient verarbeiten zu können)
- **Grafikkarte**: NVIDIA GPU mit mindestens 4 GB VRAM (z. B. NVIDIA GTX 1050 oder besser); für schnelleres Training wird eine neuere GPU mit CUDA-Unterstützung (z. B. RTX-Serie) empfohlen
- **Speicherplatz**: 10 GB oder mehr, je nach Größe der Trainingsdaten und des Modells
- **Betriebssystem**: Windows 10 oder höher, macOS, oder eine Linux-Distribution mit Python 3.8+ und CUDA-Treiber für GPU-Beschleunigung

**Hinweis**: Ohne GPU kann das Training auf der CPU durchgeführt werden, allerdings wird es deutlich langsamer sein und mehrere Stunden für einfache Datensätze und kleinere Trainingsläufe dauern.


### Schritte zum Training
Um das Modell anzupassen oder zu erweitern:
1. In den JSON-Dateien werden zusätzliche Informationen hinzugefügt.
2. Das Modell wird trainiert, indem der folgende Befehl ausgeführt wird:
```bash
python train_model.py
```
Der Pfad `fine_tuned_model` wird erstellt und speichert das trainierte Modell.

3. Trainingseinstellungen wie num_train_epochs und per_device_train_batch_size können in train_model.py angepasst werden.


### Ergebnisse
1. Ergebnis: 04.11.2024
- **Trainingszeit**: Etwa 15 Minuten pro Epoche mit den folgenden Parametern:
    - num_train_epochs=3
    - per_device_train_batch_size=4
- **Prozessor**: Intel Core i5-13500H
- **RAM**: 16 GB
- **Grafikkarte**: NVIDIA GeForce RTX 3050 (6 GB GDDR6)
- **Speicherplatz**: 4 GB
- **Betriebssystem**: Windows 11

<details>
<summary>Beispiel-Chat</summary>

Du: hi  
Bot: Hallo! Wie kann ich Ihnen helfen?

Du: hiii  
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: hii  
Bot: Hallo! Wie kann ich Ihnen helfen?

Du: Lieferzeit Trafo  
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: Kosten für Trafo  
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: Was kostet ein Transformator?  
Bot: Die Kosten für einen Transformator variieren je nach Typ und Anwendung. Bitte füllen Sie das Kontaktformular auf unserer Webseite aus oder rufen Sie uns direkt an.
</details> ```



## Versionshistorie

### v1.0.0 - 04.11.2024
- **Erstveröffentlichung** mit Hauptfunktionen:
  - Erstellung des Projektstrukturs
  - Erstellung der JSON Dateien (wenige FAQ, Dialoge und Regeln)
  - **Synonym- und Fuzzy-Matching** für ähnliche Anfragen
  - **Retrieval-Augmented Generation (RAG)** für dynamische Antworten


## Lizenz
Dieses Projekt unterliegt der MIT-Lizenz. Siehe die LICENSE-Datei für weitere Details.

## Quellen
Hugging Face Transformers: Dokumentation und Nutzung des transformers-Frameworks.
LangChain: Framework für dialogbasierte Systeme und Retrieval-Augmented Generation (RAG).
FAISS by Facebook AI Research: Bibliothek für effiziente Ähnlichkeitssuche mit Vektor-Embedding.
FuzzyWuzzy: Bibliothek für Fuzzy Matching in Python.

## Haftungsausschluss
Dieses Projekt ist ein Forschungs- und Entwicklungsprojekt. Es wird ohne jegliche Garantie bereitgestellt. Alle Informationen sind nach bestem Wissen erstellt, und es wird keine Haftung für den Einsatz oder die Ergebnisse übernommen.