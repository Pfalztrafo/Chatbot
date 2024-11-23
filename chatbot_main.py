from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from modules.recommendation import get_advanced_recommendation
from utils import format_output, preprocess_text
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from fuzzywuzzy import fuzz, process
import json
import torch
import os
from datetime import datetime
import nltk
from nltk.corpus import wordnet as wn
from utils import MODEL_NAME
from modules.dialogue_manager import get_faq_answer
from langdetect import detect, DetectorFactory



faq_data = None  # Initialisierung außerhalb der Funktion


# WordNet-Daten einmalig herunterladen
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Optional für zusätzliche Sprachdaten


# Funktion, um deutsche Synonyme aus openthesaurus.txt zu laden
def load_openthesaurus_text(filepath="data/openthesaurus.txt"):
    synonyms_dict = {}
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                synonyms = line.strip().split(";")
                for word in synonyms:
                    synonyms_dict[word] = synonyms
    except FileNotFoundError:
        print(f"[DEBUG] Datei {filepath} nicht gefunden. Synonyme werden nicht geladen.")
    return synonyms_dict

# Globale Initialisierung von german_synonyms_dict
german_synonyms_dict = load_openthesaurus_text()

# Spracheinstellung (Standard: Deutsch)
language = "de"

# Dynamische Pfade für JSON-Dateien
def get_file_path(file_type, language="de"):
    """
    Gibt den Dateipfad basierend auf dem Typ und der Sprache zurück.
    """
    file_mapping = {
        "dialogues": f"data/dialogues_{language}.json",
        "decision_rules": f"data/decision_rules_{language}.json",
        "decision_trees": f"data/decision_trees_{language}.json",
        "fallback_responses": f"data/fallback_responses_{language}.json"
    }
    return file_mapping.get(file_type)



# Kategorie erkennen
def detect_category(user_input):
    user_input = user_input.lower()
    if any(keyword in user_input for keyword in ["wartung", "service", "reparatur", "inspektion", "austausch", "reinigung"]):
        return "Service"
    elif any(keyword in user_input for keyword in ["kaufen", "angebot", "verfügbarkeit", "lieferung", "produkt", "preis"]):
        return "Kaufberatung"
    elif any(keyword in user_input for keyword in ["spannung", "leistung", "technisch", "transformator", "typ", "spezifikation", "kva", "mva", "anschluss"]):
        return "Technik"
    else:
        return "Allgemein"

# Fallback-Antwort laden
def load_fallback_responses():
    with open(get_file_path("fallback_responses"), "r", encoding="utf-8") as file:
        return json.load(file)

def fallback_response(user_input):
    responses = load_fallback_responses()
    category = detect_category(user_input)
    return responses.get(category, responses["Fallback"])

#----------------------------
# Dynamischer Pfad für das Modell basierend auf der Sprache
def get_model_path():
    return f"./fine_tuned_model_{language}"

# Überprüfen, ob das Modell existiert, und das richtige Modell laden
def load_model_and_tokenizer():
    model_path = get_model_path()
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print(f"Verwende das feingetunte Modell für Sprache: {language}.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print(f"Kein feingetuntes Modell für {language} gefunden. Verwende das Standardmodell.")
    return model, tokenizer


# Wissensbasis für RAG vorbereiten
def setup_retriever(knowledge_base):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    faiss_index = FAISS.from_documents(knowledge_base, embeddings)
    retriever = faiss_index.as_retriever()
    return retriever


#----------------------------
DetectorFactory.seed = 0  # Für reproduzierbare Ergebnisse


def set_language(user_input):
    """
    Sprachwechsel basierend auf expliziten Nutzeranfragen:
    - Wechsel zu Englisch nur bei "Can you speak English".
    - Wechsel zu Deutsch nur bei "Ich möchte in Deutsch schreiben".
    - Standard: Deutsch bleibt die Sprache.
    """
    global language, model, tokenizer, hf_pipeline, llm, knowledge_base, retriever

    if "can you speak english" in user_input.lower():
        language = "en"
        print("Sprache umgestellt: Englisch")
    elif "ich möchte in deutsch schreiben" in user_input.lower():
        language = "de"
        print("Sprache umgestellt: Deutsch")
    else:
        # Keine Änderung der Sprache
        return

    # Modell und Pipeline basierend auf der aktuellen Sprache neu laden
    model_path = f"./fine_tuned_model_{language}"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Wissensdatenbank und Retriever neu laden
    knowledge_base = load_knowledge_base()
    retriever = setup_retriever(knowledge_base)


# Wissensdatenbank laden
def load_knowledge_base(language="de"):
    """
    Lädt die Dialogdaten aus der JSON-Datei und bereitet sie für die Suche vor.
    """
    file_path = get_file_path("dialogues", language)  # Korrigierter Aufruf mit Sprache
    with open(file_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    documents = []
    for item in raw_data:
        # Hauptfrage hinzufügen
        documents.append(Document(page_content=item["question"], metadata={"answer": item["answer"]}))
        
        # Synonyme hinzufügen
        if "synonyms" in item:
            for synonym in item["synonyms"]:
                documents.append(Document(page_content=synonym, metadata={"answer": item["answer"]}))

    print(f"[DEBUG] FAQ-Daten geladen: {len(documents)} Einträge")
    return documents






# Globale Initialisierung
device = 0 if torch.cuda.is_available() else -1
model, tokenizer = load_model_and_tokenizer()
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
knowledge_base = load_knowledge_base()
retriever = setup_retriever(knowledge_base)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")




# FAQ-Daten basierend auf der Sprache laden
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from utils import preprocess_text
import json

# FAQ-Daten basierend auf der Sprache laden
def load_faq_data(language="de"):
    """
    Lädt die FAQ-Daten, erweitert sie mit Synonymen und erstellt einen FAISS-Index.
    Diese Funktion initialisiert das Embeddings-Modell nur einmal und verwendet
    globale Variablen für die FAQ-Daten und den Index.
    """
    global faq_data, faq_index, embeddings_model

    # Initialisiere das Embeddings-Modell nur, wenn es nicht existiert oder None ist
    if "embeddings_model" not in globals() or embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("[DEBUG] Embeddings-Modell initialisiert.")

    # Lade die FAQ-Daten aus der JSON-Datei
    faq_data = []
    file_path = f"data/dialogues_{language}.json"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        # Konvertiere die FAQ-Daten in Documents für FAISS
        for item in raw_data:
            # Hauptfrage als Dokument hinzufügen
            faq_data.append(Document(page_content=preprocess_text(item["question"]),
                                      metadata={"answer": item["answer"]}))
            
            # Füge Synonyme hinzu, falls vorhanden
            if "synonyms" in item:
                for synonym in item["synonyms"]:
                    faq_data.append(Document(page_content=preprocess_text(synonym),
                                              metadata={"answer": item["answer"]}))
        print(f"[DEBUG] FAQ-Daten geladen: {len(faq_data)} Einträge")
    except FileNotFoundError:
        print(f"[ERROR] FAQ-Datei nicht gefunden: {file_path}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] Fehler beim Lesen der FAQ-Datei: {e}")

    # Erstelle einen FAISS-Index für die FAQ-Daten
    faq_index = FAISS.from_documents(faq_data, embeddings_model)
    print("[DEBUG] FAISS-Index erstellt.")



# Fuzzy Matching und Embeddings
def get_faq_answer_fuzzy(user_input):
    """
    Fuzzy Matching und Embeddings-basierte Suche für FAQ-Antworten.
    Diese Funktion berücksichtigt Synonyme aus der JSON-Datei und verwendet Fuzzy Matching.
    Falls keine passende Antwort gefunden wird, wird auf Embeddings-basierte Suche zurückgegriffen.
    """
    user_input = preprocess_text(user_input)  # Eingabe vorverarbeiten
    question_variants = []
    question_to_answer = {}

    # FAQ-Fragen und Synonyme in Variantenliste einfügen
    for item in faq_data:
        # Hauptfrage hinzufügen
        question_variants.append(item.page_content)
        question_to_answer[item.page_content] = item.metadata["answer"]

        # Synonyme aus JSON hinzufügen
        if "synonyms" in item.metadata:
            synonyms = item.metadata["synonyms"]
            for synonym in synonyms:
                question_variants.append(synonym)
                question_to_answer[synonym] = item.metadata["answer"]

    # Debugging: Zeige, welche Varianten für das Matching verwendet werden
    #print(f"[DEBUG] Anzahl der Matching-Varianten: {len(question_variants)}")
    #print(f"[DEBUG] Eingabe: {user_input}")

    # Fuzzy Matching anwenden
    best_match, score = process.extractOne(user_input, question_variants, scorer=fuzz.token_sort_ratio)

    # Debugging: Ergebnis des Fuzzy Matchings anzeigen
    #print(f"[DEBUG] Beste Übereinstimmung: {best_match} mit Score: {score}")

    # Wenn der Score über dem Schwellenwert liegt, Rückgabe der Antwort
    if score > 80:  # Schwellenwert anpassen, falls nötig
        return question_to_answer[best_match]

    # Fallback: Embeddings-basierte Suche, falls Fuzzy Matching keine gute Übereinstimmung findet
    #print(f"[DEBUG] Fuzzy Matching hat keine ausreichende Übereinstimmung gefunden. Fallback auf Embeddings.")
    return search_faq_with_embeddings(user_input)






# Embeddings-Suche
def search_faq_with_embeddings(query):
    """
    Suche nach der besten Übereinstimmung basierend auf Embeddings.
    """
    try:
        embedding = embeddings_model.embed_query(query)
        results = faq_index.similarity_search_by_vector(embedding, k=1)
        if results and results[0].metadata.get("answer"):
            return results[0].metadata["answer"]
        return None
    except Exception as e:
        print(f"[DEBUG] Fehler bei der Embeddings-Suche: {e}")
        return None




# Chat-Logs speichern
def save_chat_to_txt(user_message, bot_response, user_ip="Unbekannt", username="Unbekannt", folder="chat_logs"):
    os.makedirs(folder, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(folder, f"{date_str}_chat_log.txt")
    with open(filename, "a", encoding="utf-8") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] [IP: {user_ip}] [User: {username}] {user_message}\n")
        file.write(f"[{timestamp}] [Server] [Bot] {bot_response}\n")


def save_unanswered_question(user_message, filename="data/unanswered_questions.json"):
    question_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": user_message,
        "category": detect_category(user_message),
        "answer": ""
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Verzeichnisse erstellen, falls nicht vorhanden
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []  # Leere Liste, falls Datei nicht existiert oder ungültig ist
    data.append(question_data)
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)



# Haupt-Chat-Funktion
def chat():
    print("Starte den Chat (zum Beenden 'exit' eingeben)")
    user_ip = "192.168.1.10"
    username = "JohnDoe"
    load_faq_data()  # Daten initial laden

    while True:
        user_input = input("Du: ").strip()
        if not user_input:
            print("[DEBUG] Leere Eingabe erkannt. Bitte geben Sie eine Frage ein.")
            continue
        if user_input.lower() == "exit":
            print("Chat beendet.")
            break

        set_language(user_input)
        load_faq_data()  # FAQ-Daten neu laden, wenn Sprache gewechselt wird
        user_input = preprocess_text(user_input)
        category = detect_category(user_input)
        fallback_responses = load_fallback_responses()

        if category == "Service":
            response = fallback_responses.get("Service", fallback_responses["Fallback"])
        elif category == "Technik":
            response = get_advanced_recommendation(user_input, {}, language) or fallback_responses.get("Technik")
        else:
            response = get_faq_answer_fuzzy(user_input) or fallback_responses["Fallback"]

        # Unbeantwortete Frage speichern, wenn keine Antwort gefunden wird
        if not response or response == fallback_responses["Fallback"]:
            print("[DEBUG] Unanswered question detected. Saving...")
            save_unanswered_question(user_input, "data/unanswered_questions.json")

        print(format_output(response))
        save_chat_to_txt(user_input, response, user_ip=user_ip, username=username)


if __name__ == "__main__":
    load_faq_data()  # Daten initial laden
    chat()
