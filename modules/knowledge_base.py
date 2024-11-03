import json
from utils import preprocess_text

def load_trafo_info():
    """Lädt allgemeine Informationen über Transformatoren aus der Datei 'trafo_info.json'."""
    with open("data/trafo_info.json", "r", encoding="utf-8") as file:
        return json.load(file)

def get_info_by_topic(topic):
    """Sucht nach einem spezifischen Thema in den Transformator-Informationen und gibt die entsprechende Information zurück."""
    data = load_trafo_info()
    topic = preprocess_text(topic)
    for item in data:
        if preprocess_text(item["topic"]) == topic:
            return item["info"]
    return "Keine Informationen zu diesem Thema gefunden."

def load_services_info():
    """Lädt Informationen zu den Dienstleistungen von Pfalztrafo aus der Datei 'pfalztrafo_services.json'."""
    with open("data/pfalztrafo_services.json", "r", encoding="utf-8") as file:
        return json.load(file)

def get_service_description(service):
    """Gibt eine Beschreibung der angefragten Dienstleistung zurück, wenn vorhanden."""
    data = load_services_info()
    service = preprocess_text(service)
    for item in data:
        if preprocess_text(item["service"]) == service:
            return item["description"]
    return "Service nicht gefunden."

# Erweiterung: Wartungsinformationen
def get_maintenance_info():
    """Gibt alle Informationen zu Wartungsservices und Anforderungen zurück, falls vorhanden."""
    data = load_trafo_info()
    maintenance_info = []
    for item in data:
        if "wartung" in preprocess_text(item["topic"]):  # Filter für wartungsbezogene Themen
            maintenance_info.append(item["info"])
    return " ".join(maintenance_info) if maintenance_info else "Es liegen keine spezifischen Wartungsinformationen vor."
