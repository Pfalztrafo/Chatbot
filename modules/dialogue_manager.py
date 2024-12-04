import json
from utils import preprocess_text
from datetime import datetime

# Funktion zum Laden von Dialogen (FAQs)
def load_faq():
    """
    Lädt FAQ-Daten aus der JSON-Datei.
    """
    filename = "data/faq_sales.json"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Fehler: Datei {filename} nicht gefunden.")
        return []

# Antwort auf eine FAQ-Frage basierend auf Matching
def get_faq_answer(question):
    """
    Gibt eine Antwort auf eine FAQ-Frage zurück, falls eine Übereinstimmung gefunden wird.
    """
    faq_data = load_faq()
    processed_question = preprocess_text(question)

    for item in faq_data:
        # Exakte Übereinstimmung prüfen
        if preprocess_text(item["question"]) == processed_question:
            return item["answer"]

        # Synonyme prüfen
        if "synonyms" in item:
            for synonym in item["synonyms"]:
                if preprocess_text(synonym) == processed_question:
                    return item["answer"]

    # Keine Antwort gefunden
    return None

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
