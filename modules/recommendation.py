import json
from utils import preprocess_text

# Funktion zum Laden der Entscheidungsbäume
def load_decision_trees():
    """
    Lädt den Entscheidungsbaum aus der JSON-Datei.
    """
    filename = "data/decision_trees.json"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Fehler: Datei {filename} nicht gefunden.")
        return {}

# Dynamisches Empfehlungssystem basierend auf Entscheidungsbaum
def get_advanced_recommendation(application, parameters, user_input):
    """
    Gibt eine Empfehlung basierend auf der Auswahl der Anwendung und den benutzerspezifischen Parametern.
    """
    if should_abort(user_input):
        return "Die Beratung wurde abgebrochen. Wie kann ich sonst helfen?"

    trees = load_decision_trees()
    application = preprocess_text(application)

    if application in trees:
        app_data = trees[application]
        
        # Durchlaufe den Entscheidungsbaum
        return traverse_decision_tree(app_data["options"], parameters, user_input)
    
    return "Keine spezifische Empfehlung für Ihre Anwendung gefunden."

# Abbruchprüfung
def should_abort(user_input):
    """
    Überprüft, ob der Benutzer den Entscheidungsbaum verlassen möchte.
    """
    exit_phrases = ["abbrechen", "exit", "zurück", "stopp", "nein"]
    return preprocess_text(user_input).lower() in exit_phrases

# Rekursive Funktion zur Durchlauf des Entscheidungsbaums
def traverse_decision_tree(options, parameters, user_input):
    """
    Durchläuft den Entscheidungsbaum rekursiv und gibt eine Antwort zurück.
    """
    for option, option_data in options.items():
        # Frage des aktuellen Knotens
        question = option_data["question"]
        
        # Prüfen, ob der Benutzer abbrechen möchte
        if should_abort(user_input):
            return "Die Beratung wurde abgebrochen. Wie kann ich sonst helfen?"
        
        # Wenn es Optionen gibt, frage weiter
        if "options" in option_data:
            user_value = parameters.get(option)
            if user_value is not None:
                if user_value in option_data["options"]:
                    return traverse_decision_tree(option_data["options"][user_value]["options"], parameters, user_input)
                else:
                    return option_data.get("response", "Keine passende Option gefunden.")
        
        # Fallback-Antwort für den Endpunkt
        if "response" in option_data:
            return option_data["response"]

    return "Es tut mir leid, es konnte keine passende Empfehlung gefunden werden."

# Prüfung für Beratungskategorie und Folgefrage
def check_for_advisory_category(category):
    """
    Fügt eine Frage hinzu, ob der Benutzer eine Beratung starten möchte.
    """
    if category == "Beratung":
        return "Sollen wir mit einer Beratung für den Kauf eines Transformators fortfahren? Bitte antworten Sie mit Ja oder Nein."
    return None

# Verarbeitung der Benutzerantwort
def handle_user_response(user_input, category):
    """
    Handhabt die Benutzerantwort, um den Entscheidungsbaum zu starten oder abzubrechen.
    """
    if user_input.lower() in ["ja", "yes"] and category == "Beratung":
        return get_advanced_recommendation("kaufberatung", {}, user_input)
    elif user_input.lower() in ["nein", "no"]:
        return "Alles klar. Wie kann ich Ihnen sonst helfen?"
    return None
