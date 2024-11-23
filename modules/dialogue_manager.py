import json
from utils import preprocess_text, load_synonyms

# Lädt OpenThesaurus-Synonyme einmalig
OPEN_THESAURUS_SYNONYMS = load_synonyms("data/openthesaurus.txt")

# Funktion zum Laden von Dialogen basierend auf der Sprache
def load_dialogues(language="de"):
    """
    Lädt Dialoge aus der JSON-Datei basierend auf der gewählten Sprache.
    """
    filename = f"data/dialogues_{language}.json"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Fehler: Datei {filename} nicht gefunden.")
        return []

# Antwort auf eine FAQ-Frage basierend auf der Sprache
def get_faq_answer(question, language="de"):
    """
    Gibt eine Antwort auf eine FAQ-Frage basierend auf exaktem Matching und Synonymen zurück.
    """
    dialogues = load_dialogues(language)
    processed_question = preprocess_text(question)

    for item in dialogues:
        # Exakte Übereinstimmung prüfen
        if preprocess_text(item["question"]) == processed_question:
            return item["answer"]

        # Synonyme prüfen (JSON-Synonyme)
        if "synonyms" in item:
            for synonym in item["synonyms"]:
                if preprocess_text(synonym) == processed_question:
                    return item["answer"]

        # Synonyme prüfen (OpenThesaurus-Synonyme)
        for synonym in OPEN_THESAURUS_SYNONYMS.get(processed_question, []):
            if preprocess_text(item["question"]) == preprocess_text(synonym):
                return item["answer"]

    # Keine Antwort gefunden
    return None

# Suche nach verwandten FAQ-Antworten basierend auf der Sprache
def get_related_faq(question, language="de"):
    """
    Gibt eine verwandte Antwort zurück, wenn keine exakte Übereinstimmung gefunden wird.
    """
    dialogues = load_dialogues(language)
    question = preprocess_text(question)

    for item in dialogues:
        if preprocess_text(item["question"]) in question:
            return item["answer"]

    return "Entschuldigung, ich konnte keine passende Antwort finden. Möchten Sie andere Dienstleistungen kennenlernen?"
