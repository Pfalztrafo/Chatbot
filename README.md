# Support-Chatbot zur Kaufberatung für Transformatoren

Dieser Chatbot wurde entwickelt, um Kunden bei Kaufempfehlungen und allgemeinen Fragen zu Transformatoren zu unterstützen. Er bietet Informationen zu Transformator-Typen, Services vom Unternehmen Pfalztrafo und stellt Empfehlungen basierend auf spezifischen Kundenanfragen zur Verfügung.

## Inhaltsverzeichnis
- [Übersicht](#übersicht)
- [Features](#features)
- [Verwendete Technologien](#verwendete-technologien)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Training](#training)
- [Verwendung](#verwendung)
- [Versionshistorie](#versionshistorie)
- [Quellen](#quellen)
- [Lizenz und Haftungsausschluss](#lizenz-und-haftungsausschluss)


## Übersicht
Der Support-Chatbot zur Kaufberatung für Transformatoren basiert auf einem feinabgestimmten Sprachmodell (fine-tuned LLM), das auf `google/flan-t5-base` basiert. Mithilfe von Retrieval-Augmented Generation (RAG) kann der Bot dynamisch auf Kundenfragen reagieren und gibt dabei gezielte Informationen zu Transformatoren und Dienstleistungen von Pfalztrafo.

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
Die Projektstruktur ist in mehrere Module und Datenquellen unterteilt (Stand 04.11.2024):

```plaintext
├── chatbot_main.py              # Hauptskript zum Ausführen des Chatbots
├── train_model.py               # Trainingsskript zum Feinabstimmen des Modells
├── utils.py                     # Konfiguration und Hilfsfunktionen
├── data/
│   ├── openthesaurus.txt        # OpenThesaurus Textdatei mit deutschen Synonymen
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

**Bibliotheken installieren**:
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
pip install nltk

Integration Webseite
pip install fastapi uvicorn

als Exe erstellen
Probleme mit pydantic
pip uninstall pydantic
pip install pydantic==1.10.8
pip install langdetect

```


## Training
### Mindestanforderungen für das Training
Das Training des Modells erfordert deutlich höhere Systemressourcen als die Nutzung. Die folgenden Mindestanforderungen ermöglichen ein akzeptables Trainingstempo:
- **Prozessor**: Intel Core i5 oder AMD Ryzen 5 (Quad-Core oder höher)
- **RAM**: 16 GB (Mehr ist empfohlen, um Trainingsdaten effizient verarbeiten zu können)
- **Grafikkarte**: NVIDIA GPU mit mindestens 4 GB VRAM (z. B. NVIDIA GTX 1050 oder besser); für schnelleres Training wird eine neuere GPU mit CUDA-Unterstützung (z. B. RTX-Serie) empfohlen
- **Speicherplatz**: 10 GB oder mehr, je nach Größe der Trainingsdaten und des Modells
- **Betriebssystem**: Windows 10 oder höher, macOS, oder eine Linux-Distribution mit Python 3.8+ und CUDA-Treiber für GPU-Beschleunigung

**Hinweis**: Ohne GPU kann das Training auf der CPU durchgeführt werden, allerdings wird es deutlich langsamer sein und mehrere Stunden für einfache Datensätze und kleinere Trainingsläufe dauern.

### Eigene Hardware-Spezifikationen:
- **Prozessor**: Intel Core i5-13500H
- **RAM**: 16 GB
- **Grafikkarte**: NVIDIA GeForce RTX 3050 (6 GB GDDR6)
- **Speicherplatz**: 4 GB
- **Betriebssystem**: Windows 11

### Schritte zum Training
Um das Modell anzupassen oder zu erweitern:
1. In den JSON-Dateien werden zusätzliche Informationen hinzugefügt.
2. Das Modell wird trainiert, indem der folgende Befehl ausgeführt wird:
```bash
python train_model.py
```
Der Pfad `fine_tuned_model` wird erstellt und speichert das trainierte Modell.

3. Trainingseinstellungen wie num_train_epochs und per_device_train_batch_size können in train_model.py angepasst werden.

### Trainingskonfiguration
Die folgenden Parameter sind entscheidend für das Fein-Tuning des Modells und können je nach Hardware und gewünschtem Ergebnis angepasst werden:

- **output_dir**: Der Speicherort für das trainierte Modell und die Checkpoints.  
  Beispiel: `output_dir="./fine_tuned_model"` legt einen Ordner an, der während des Trainings automatisch mit Modellen und Checkpoints befüllt wird.

- **eval_strategy**: Legt fest, ob und wie oft eine Evaluation während des Trainings erfolgt.  
  Wenn `eval_strategy="no"` gesetzt ist, erfolgt keine Evaluation, was Speicher spart und das Training beschleunigt. Alternativ kann `eval_strategy="steps"` oder `"epoch"` gewählt werden, um nach jeder bestimmten Schrittanzahl bzw. Epoche eine Evaluation durchzuführen. Regelmäßige Evaluationen erhöhen jedoch die GPU-Auslastung und Trainingszeit.

- **learning_rate**: Die Lernrate steuert die Größe der Schritte, die das Modell bei jedem Update macht.  
  Typische Werte liegen zwischen `1e-5` und `5e-5`. Eine kleinere Lernrate (z. B. `2e-5`) ist für präzisere Anpassungen geeignet, kann jedoch das Training verlangsamen und erfordert möglicherweise mehr Epochen. Eine höhere Lernrate führt zu schnellerem Training, birgt jedoch das Risiko von Überanpassung (Overfitting), da größere Updates vorgenommen werden.

- **per_device_train_batch_size**: Die Anzahl der Trainingsbeispiele pro Batch, die pro Gerät verarbeitet wird.  
  Kleinere Werte (z. B. `4`) reduzieren den GPU-Speicherbedarf und sind für GPUs mit weniger VRAM geeignet, während größere Werte (z. B. `16` oder `32`) bei leistungsstarken GPUs das Training beschleunigen können. Als Faustregel gilt: Verdoppelt sich die Batch-Größe, verdoppelt sich der Speicherbedarf der GPU. Zu große Batches können jedoch zu Überanpassung führen.

- **num_train_epochs**: Anzahl der Epochen, die das Modell durchläuft, um die Daten zu lernen.  
  Für kleinere Datensätze reichen oft 1-3 Epochen, während umfangreichere Datensätze mehr Epochen erfordern können. Mehr Epochen verbessern die Genauigkeit, können jedoch die Trainingszeit und den GPU-Speicherbedarf erhöhen, da das Modell mehrfach auf die Daten zugreift.

- **weight_decay**: Ein Parameter zur Kontrolle der Modellkomplexität und zur Reduzierung von Überanpassung.  
  Typische Werte liegen zwischen `0.01` und `0.1`. Ein kleinerer Wert, wie `0.01`, bewirkt einen minimalen Gewichtszerfall, der die Modellparameter leicht reguliert, während höhere Werte die Regularisierung verstärken und Overfitting weiter eindämmen können.



### Ergebnisse
#### Ergebnis v1.0.0 vom 04.11.2024 (Eigenes Laptop)
- **Trainingseinheit**: 36 Epochen = 2 min
- **Parameter-Konfiguration**:
  - `eval_strategy="no"`: Evaluation deaktiviert, nur Training
  - `learning_rate=2e-5`: Feinabstimmungs-Lernrate für präzise Updates
  - `per_device_train_batch_size=4`: Batch-Größe pro Gerät (RTX 3050 mit 6 GB VRAM)
  - `num_train_epochs=3`: Anzahl der Epochen für bessere Modellanpassung
  - `weight_decay=0.01`: Gewichtszerfall zur Reduzierung von Overfitting

<details>
<summary>Beispiel-Chat-04.11.2024</summary>

Du: heeyy   
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: hiii   
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: hii   
Bot: Hallo! Wie kann ich Ihnen helfen?

Du: Was für Arten von Trafos habt ihr?   
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: Was bietet ihr an   
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: Arten von Trafos   
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: Welche Transformator-Typen gibt es?   
Bot: Es gibt verschiedene Typen, darunter Leistungstransformatoren und Verteiltransformatoren.

Du: Transformator-Typen   
Bot: Es gibt verschiedene Typen, darunter Leistungstransformatoren und Verteiltransformatoren.

Du: Ülprobe   
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: Lebensdauer eines Transformators   
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: Leistungstransformator   
Bot: Ich habe leider keine Antwort auf diese Frage.

Du: Zertifikate für Trafos   
Bot: Ich habe leider keine Antwort auf diese Frage.
</details> ```

#### Ergebnis v1.0.0 vom 04.11.2024 (Eigenes Laptop)
- **Trainingseinheit**: 6 Epochen = 20 Sekunden
- **Parameter-Konfiguration**:
  - `eval_strategy="no"`: Evaluation deaktiviert, nur Training
  - `learning_rate=2e-5`: Feinabstimmungs-Lernrate für präzise Updates
  - `per_device_train_batch_size=4`: Batch-Größe pro Gerät (RTX 3050 mit 6 GB VRAM)
  - `num_train_epochs=3`: Anzahl der Epochen für bessere Modellanpassung
  - `weight_decay=0.01`: Gewichtszerfall zur Reduzierung von Overfitting

<details>
<summary>Beispiel-Chat-04.11.2024</summary>

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



## Versionshistorie

### v1.1.0 - 07.11.2024
- **Speicherung des Chatverlaufs**: Tägliche Protokollierung von Nutzer- und Bot-Nachrichten zur Nachverfolgung von Interaktionen.
- **Speicherung der Trainingsdaten und -dauer**: Dokumentation der Trainingszeit, Verlustwerte, Epochenanzahl und Geräte-Spezifikationen für Transparenz und Wiederholbarkeit.
- **Speicherung unbeantworteter Fragen**: Erfassung von Nutzeranfragen ohne Antwort für eine gezielte Erweiterung der Wissensbasis.
- **Erweiterte Fallback-Strategie** für unbeantwortete Fragen: Dynamische Erkennung und Weiterleitung an relevante Kategorien oder Services.
- **Kategorieerkennung und Fallback-Antwort**: Verbesserte Kategorisierung von Fragen zur optimierten Benutzerführung.
- **Multilinguale Unterstützung** (Deutsch und Englisch): Erstellung separater JSON-Dateien für FAQs, Dialoge und Entscheidungstabellen in beiden Sprachen.
- **Synonymunterstützung und Fuzzy Matching**: Einbindung von WordNet und OpenThesaurus für bessere Abdeckung ähnlicher Anfragen in Deutsch und Englisch.
- **Struktur für nachhaltige Datenspeicherung**: Optimierte Ordnerstruktur und Versionierung zur langfristigen Wartung und Erweiterung.

### v1.0.0 - 04.11.2024
- **Erstveröffentlichung** mit Hauptfunktionen:
  - Erstellung des Projektstrukturs
  - Erstellung der JSON Dateien (FAQ, Dialoge und Regeln)
  - **Synonym- und Fuzzy-Matching** für ähnliche Anfragen
  - **Retrieval-Augmented Generation (RAG)** für dynamische Antworten



## Quellen
- **LLM (Large Language Model)**: Das Projekt verwendet das vortrainierte Modell `google/flan-t5-base`, [Link](https://huggingface.co/google/flan-t5-base/)
- **LangChain**: Eine Framework zur Entwicklung von Anwendungen mit großen Sprachmodellen. [Link](https://pypi.org/project/langchain/)
- **Hugging Face Transformers**: Eine Bibliothek für die Arbeit mit vortrainierten Transformer-Modellen. [Link](https://pypi.org/project/transformers/) [Link](https://huggingface.co/docs/transformers/main_classes/trainer)
- **FAISS**: Eine Bibliothek für effiziente Ähnlichkeitssuche und Clustering von dichten Vektoren. [Link](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
- **FuzzyWuzzy**: Ein Python-Paket für Fuzzy-String-Matching. [Link](https://pypi.org/project/fuzzywuzzy/)
- **LangChain Tutorial**: Ein Crash-Kurs zur Verwendung von LangChain in Python. [Link](https://www.python-engineer.com/posts/langchain-crash-course/)
- **FAISS und Sentence-Transformers in 5 Minuten**: Ein Leitfaden zur schnellen Implementierung von FAISS mit Sentence-Transformers. [Link](https://www.stephendiehl.com/posts/faiss/)
- **LangChain FAISS Dokumentation**: Die offizielle Dokumentation zur Integration von FAISS in LangChain. [Link](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
- **LangChain C Transformers**: Informationen zur Verwendung der C Transformers-Bibliothek innerhalb von LangChain. [Link](https://python.langchain.com/docs/integrations/providers/ctransformers/)




## Lizenz und Haftungsausschluss

Copyright (c) 2024 Anonym

Dieses Projekt ist ausschließlich für die Überprüfung durch die Hochschule Kaiserslautern und zur Einsicht durch die zugewiesenen Prüfer gedacht. Die Nutzung, Kopie, Veränderung und Verbreitung ist nur mit ausdrücklicher Genehmigung des Autors gestattet. Jegliche kommerzielle Nutzung oder Verbreitung ohne Genehmigung ist untersagt.

Die Software wird "wie besehen" bereitgestellt, ohne jegliche ausdrückliche oder stillschweigende Garantie, einschließlich und ohne Einschränkung der Garantien zur Gebrauchstauglichkeit, Eignung für einen bestimmten Zweck und Nichtverletzung. Der Autor übernimmt keine Haftung für direkte, indirekte, zufällige oder Folgeschäden, die aus der Nutzung oder Unfähigkeit zur Nutzung der Software entstehen.

**Hinweis**: Dieses Projekt ist ein Forschungs- und Entwicklungsprojekt und nicht für die öffentliche Nutzung oder den produktiven Einsatz bestimmt. Alle Informationen wurden nach bestem Wissen erstellt, aber es wird keine Garantie für die Vollständigkeit, Richtigkeit oder Aktualität der bereitgestellten Informationen übernommen.