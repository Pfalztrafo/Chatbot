import torch

# Konfiguration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/flan-t5-base"

# Hilfsfunktionen
def preprocess_text(text):
    return text.strip().lower()

def format_output(response):
    return f"Bot: {response}"

# utils.py
def load_synonyms(file_path="data/openthesaurus.txt"):
    """
    Lädt Synonyme aus der OpenThesaurus-Datei und erstellt ein Wörterbuch.
    """
    synonyms_dict = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # Ignoriere Kommentare und leere Zeilen
                if line.startswith("#") or not line.strip():
                    continue
                
                # Teile die Synonyme und füge sie ins Wörterbuch ein
                synonyms = line.strip().split(";")
                for word in synonyms:
                    word = word.lower().strip()
                    synonyms_dict[word] = set(synonyms) - {word}
        return synonyms_dict
    except FileNotFoundError:
        print(f"Fehler: Datei {file_path} nicht gefunden.")
        return {}
