# Support-Chatbot im Transformatorensegment

Dieser Chatbot wurde entwickelt, um Kunden bei Kaufempfehlungen und allgemeinen Fragen zu Transformatoren zu unterstützen. Er bietet Informationen zu Transformator-Typen, Services vom Unternehmen Pfalztrafo und stellt Empfehlungen basierend auf spezifischen Kundenanfragen zur Verfügung.

## Inhaltsverzeichnis
- [Übersicht](#übersicht)
- [Features](#features)
- [Verwendete Technologien](#verwendete-technologien)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Training des Modells](#training-des-modells)
- [Anpassungen und Training](#anpassungen-und-training)
- [Lizenz](#lizenz)
- [Haftungsausschluss](#haftungsausschluss)

## Übersicht
Der Transformer Support Chatbot basiert auf einem feinabgestimmten Sprachmodell (fine-tuned LLM), das auf `google/flan-t5-base` basiert. Mithilfe von Retrieval-Augmented Generation (RAG) kann der Bot dynamisch auf Kundenfragen reagieren und gibt dabei gezielte Informationen zu Transformatoren und Dienstleistungen von Pfalztrafo.

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
├── chatbot_main.py           # Hauptskript zum Ausführen des Chatbots
├── train_model.py            # Trainingsskript zum Feinabstimmen des Modells
├── data/
│   ├── dialogues.json        # Häufig gestellte Fragen und Antworten
│   ├── trafo_info.json       # Allgemeine Infos über Transformatoren
│   ├── pfalztrafo_services.json # Dienstleistungen von Pfalztrafo
│   ├── rules.json            # Entscheidungsregeln für Empfehlungen
├── modules/
│   ├── dialogue_manager.py   # Modul zur Verwaltung von Dialogen
│   ├── knowledge_base.py     # Modul für den Zugriff auf Transformatoren-Wissen
│   ├── recommendation.py     # Modul für Kaufempfehlungen und Entscheidungslogik
├── utils/
│   ├── config.py             # Konfigurationsdatei
│   ├── helpers.py            # Hilfsfunktionen
├── fine_tuned_model/         # Verzeichnis für das trainierte Modell
└── README.md                 # Diese Dokumentation

## Installation

1. **Abhängigkeiten installieren**:
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



Mit diesen Anweisungen ab "Installation" enthält die `README.md` alle notwendigen Schritte, um das Projekt einzurichten, zu verwenden und zu erweitern.
