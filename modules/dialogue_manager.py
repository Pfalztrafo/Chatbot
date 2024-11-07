import json
from utils import preprocess_text

# Funktion zum Laden von Dialogen basierend auf der Sprache
def load_dialogues(language="de"):
    filename = f"data/dialogues_{language}.json"
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

# Antwort auf eine FAQ-Frage basierend auf der Sprache
def get_faq_answer(question, language="de"):
    dialogues = load_dialogues(language)
    question = preprocess_text(question)

    for item in dialogues:
        if preprocess_text(item["question"]) in question:
            return item["answer"]
        if "synonyms" in item:
            for synonym in item["synonyms"]:
                if preprocess_text(synonym) in question:
                    return item["answer"]

    return "Ich habe leider keine Antwort auf diese Frage."

# Suche nach verwandten FAQ-Antworten basierend auf der Sprache
def get_related_faq(question, language="de"):
    dialogues = load_dialogues(language)
    question = preprocess_text(question)

    for item in dialogues:
        if preprocess_text(item["question"]) in question:
            return item["answer"]

    return "Entschuldigung, ich konnte keine passende Antwort finden. Möchten Sie andere Dienstleistungen kennenlernen?"
