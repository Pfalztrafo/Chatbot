import json
from utils import preprocess_text

def load_rules():
    with open("data/rules.json", "r", encoding="utf-8") as file:
        return json.load(file)

def get_recommendation(condition):
    rules = load_rules()
    condition = preprocess_text(condition)
    for rule in rules:
        if preprocess_text(rule["condition"]) in condition:
            return rule["recommendation"]
    return "Ich habe leider keine passende Empfehlung gefunden."

# Erweiterung: Empfehlungen basierend auf Anwendung und Spannung
def get_advanced_recommendation(application, voltage):
    if "industrie" in preprocess_text(application):
        if voltage > 10000:
            return "Ein Leistungstransformator wäre ideal für Ihre Anwendung."
        else:
            return "Ein Verteiltransformator ist für industrielle Anwendungen unter 10 kV geeignet."
    elif "privat" in preprocess_text(application):
        return "Für private Anwendungen empfehlen wir einen kompakten Verteiltransformator."
    return "Keine spezifische Empfehlung für Ihre Anwendung gefunden."
