import json
from utils import preprocess_text

# Funktion zum Laden der Entscheidungsregeln basierend auf der Sprache
def load_decision_rules(language="de"):
    filename = f"data/decision_rules_{language}.json"
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

# Erweitertes Empfehlungssystem basierend auf Anwendung und Bedingungen
def get_advanced_recommendation(application, parameters, language="de"):
    rules = load_decision_rules(language)
    application = preprocess_text(application)

    if application in rules:
        app_data = rules[application]
        
        # Durchlaufe alle Bedingungen und wende passende Empfehlungen an
        for condition in app_data.get("conditions", []):
            param = condition["parameter"]
            threshold = condition["threshold"]
            user_value = parameters.get(param)
            
            if user_value is not None:
                if user_value > threshold:
                    return condition["recommendation_above"]
                else:
                    return condition["recommendation_below"]
        
        # Fallback auf die Standardempfehlung, falls keine Bedingung zutrifft
        if "default_recommendation" in app_data:
            return app_data["default_recommendation"]

    return "Keine spezifische Empfehlung für Ihre Anwendung gefunden."
