import json
from utils import preprocess_text

def load_faq():
    with open("data/dialogues.json", "r", encoding="utf-8") as file:
        return json.load(file)

def get_faq_answer(question):
    faq_data = load_faq()
    question = preprocess_text(question)

    for item in faq_data:
        # Prüfen auf exakte Frage
        if preprocess_text(item["question"]) in question:
            return item["answer"]

        # Prüfen auf Synonyme, falls vorhanden
        if "synonyms" in item:
            for synonym in item["synonyms"]:
                if preprocess_text(synonym) in question:
                    return item["answer"]

    return "Ich habe leider keine Antwort auf diese Frage."

# Erweiterung: Suche nach verwandten Fragen für bessere Vorschläge
def get_related_faq(question):
    faq_data = load_faq()
    question = preprocess_text(question)
    
    for item in faq_data:
        if preprocess_text(item["question"]) in question:
            return item["answer"]

    return "Entschuldigung, ich konnte keine passende Antwort finden. Möchten Sie andere Dienstleistungen kennenlernen?"