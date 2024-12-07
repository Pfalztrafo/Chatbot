import json
from utils import preprocess_text
from datetime import datetime
from fuzzywuzzy import fuzz

# Funktion zum Laden von Dialogen (FAQs)
def load_faq(file_path=None):
    """
    Lädt FAQ-Daten aus einer spezifischen JSON-Datei oder kombiniert mehrere Dateien.
    Wenn file_path angegeben ist, wird nur diese Datei geladen.
    """
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Fehler: Datei {file_path} nicht gefunden.")
        except json.JSONDecodeError as e:
            print(f"Fehler beim Laden der Datei {file_path}: {e}")
        return []

    # Standard: Alle Dateien kombinieren
    filenames = ["data/faq_sales.json", "data/faq_general.json"]
    combined_faq = []

    for filename in filenames:
        try:
            with open(filename, "r", encoding="utf-8") as file:
                combined_faq.extend(json.load(file))
        except FileNotFoundError:
            print(f"Fehler: Datei {filename} nicht gefunden.")
        except json.JSONDecodeError as e:
            print(f"Fehler beim Laden der Datei {filename}: {e}")

    return combined_faq




# Antwort auf eine FAQ-Frage basierend auf Matching
def get_faq_answer(question, category=None, threshold=70):
    """
    Gibt eine Antwort auf eine FAQ-Frage basierend auf Fuzzy Matching zurück.
    Berücksichtigt optional eine Kategorie.
    """
    file_path = "data/faq_sales.json" if category == "Vertrieb" else "data/faq_general.json"
    faq_data = load_faq(file_path)

    # Filtern nach Kategorie
    if category:
        faq_data = [item for item in faq_data if item.get("category") == category]
        print(f"[DEBUG] Anzahl gefilterter Einträge für Kategorie '{category}': {len(faq_data)}")

    best_match = None
    best_score = 0
    processed_question = preprocess_text(question)

    for item in faq_data:
        score = fuzz.ratio(processed_question, preprocess_text(item["question"]))
        if score > threshold and score > best_score:
            best_match = item["answer"]
            best_score = score

    return best_match, best_score






# Fallback: Nicht beantwortete Fragen speichern
def save_unanswered_question(question):
    """
    Speichert nicht beantwortete Fragen in einer JSON-Datei.
    """
    filename = "data/unanswered_questions.json"
    unanswered_data = {
        "question": question,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(unanswered_data)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

# Suche nach verwandten FAQ-Antworten
def get_related_faq(question):
    """
    Gibt verwandte FAQ-Antworten zurück, falls keine exakte Übereinstimmung gefunden wird.
    """
    faq_data = load_faq()
    processed_question = preprocess_text(question)

    for item in faq_data:
        if processed_question in preprocess_text(item["question"]):
            return item["answer"]

    # Fallback
    return "Entschuldigung, ich konnte keine passende Antwort finden. Bitte stellen Sie eine andere Frage."
