import json
from utils import preprocess_text
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Embeddings-Modell laden und FAQ-Daten in beiden Sprachen verarbeiten
def load_faq_embeddings():
    documents = []

    # Deutsch FAQ-Daten laden
    with open("data/dialogues_de.json", "r", encoding="utf-8") as file:
        faq_data_de = json.load(file)
    documents.extend([
        Document(page_content=item["question"], metadata={"answer": item["answer"]})
        for item in faq_data_de
    ])

    # Englisch FAQ-Daten laden
    with open("data/dialogues_en.json", "r", encoding="utf-8") as file:
        faq_data_en = json.load(file)
    documents.extend([
        Document(page_content=item["question"], metadata={"answer": item["answer"]})
        for item in faq_data_en
    ])

    # Verwende ein Embeddings-Modell, um die Texte zu vektorisieren
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # FAISS-Index mit den Dokumenten erstellen
    faq_index = FAISS.from_documents(documents, embeddings)
    return faq_index


# Dialoge basierend auf Sprache laden
def load_dialogues(language="de"):
    filename = f"data/dialogues_{language}.json"
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

# Informationen zu einem spezifischen Thema abrufen
def get_info_by_topic(topic, language="de"):
    dialogues = load_dialogues(language)
    topic = preprocess_text(topic)

    for item in dialogues:
        if item.get("category") == "Technik" and preprocess_text(item["question"]) == topic:
            return item["answer"]

    return "Keine Informationen zu diesem Thema gefunden."

# Beschreibung eines bestimmten Services abrufen
def get_service_description(service, language="de"):
    dialogues = load_dialogues(language)
    service = preprocess_text(service)

    for item in dialogues:
        if item.get("category") == "Service" and preprocess_text(item["question"]) == service:
            return item["answer"]

    return "Service nicht gefunden."

# Wartungsinformationen abrufen
def get_maintenance_info(language="de"):
    dialogues = load_dialogues(language)
    maintenance_info = []

    for item in dialogues:
        if item.get("category") == "Technik" and "wartung" in preprocess_text(item["question"]):
            maintenance_info.append(item["answer"])

    return " ".join(maintenance_info) if maintenance_info else "Es liegen keine spezifischen Wartungsinformationen vor."
