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





# WordNet-Daten einmalig herunterladen
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Optional für zusätzliche Sprachdaten


# Spracheinstellung (Standard: Deutsch)
language = "de"

# Dynamische Pfade für JSON-Dateien
def get_file_path(file_type):
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
def load_knowledge_base():
    with open(get_file_path("dialogues"), "r", encoding="utf-8") as file:
        raw_data = json.load(file)
    documents = [
        Document(page_content=item["question"], metadata={"answer": item["answer"]})
        for item in raw_data
    ]
    return documents



# Globale Initialisierung
device = 0 if torch.cuda.is_available() else -1
model, tokenizer = load_model_and_tokenizer()
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
knowledge_base = load_knowledge_base()
retriever = setup_retriever(knowledge_base)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")



# Fuzzy Matching und Embeddings
def get_faq_answer_fuzzy(user_input):
    user_input = preprocess_text(user_input)
    question_variants = []
    question_to_answer = {}
    for item in knowledge_base:
        question_variants.append(item.page_content)
        question_to_answer[item.page_content] = item.metadata["answer"]
    best_match, score = process.extractOne(user_input, question_variants, scorer=fuzz.token_sort_ratio)
    if score > 70:
        return question_to_answer[best_match]
    return search_faq_with_embeddings(user_input)




# Embeddings-Suche
def search_faq_with_embeddings(query):
    """
    Suche nach der besten Übereinstimmung basierend auf Embeddings.
    """
    try:
        # Sicherstellen, dass query ein String ist
        if not isinstance(query, str):
            return None  # Kein Ergebnis gefunden

        # Verwende die `invoke`-Methode statt `get_relevant_documents`
        results = retriever.invoke({"query": query})
        # results = retriever.get_relevant_documents(query)

        if results and results[0].metadata.get("answer"):
            return results[0].metadata["answer"]
        else:
            return None  # Kein Ergebnis gefunden
    except Exception:
        return None  # Kein Ergebnis gefunden




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
    while True:
        user_input = input("Du: ")
        if user_input.lower() == "exit":
            print("Chat beendet.")
            break

        set_language(user_input)
        user_input = preprocess_text(user_input)
        category = detect_category(user_input)
        fallback_responses = load_fallback_responses()

        if category == "Service":
            response = fallback_responses.get("Service", fallback_responses["Fallback"])
            print(format_output(response))
            save_chat_to_txt(user_input, response, user_ip=user_ip, username=username)
            continue

        if category == "Technik":
            response = get_advanced_recommendation(user_input, {}, language)
            if response:
                print(format_output(response))
                save_chat_to_txt(user_input, response, user_ip=user_ip, username=username)
                continue
            response = fallback_responses.get("Technik", fallback_responses["Fallback"])
            print(format_output(response))
            save_chat_to_txt(user_input, response, user_ip=user_ip, username=username)
            continue

        # Prüfen, ob FAQ eine Antwort liefert
        response = get_faq_answer_fuzzy(user_input)  # Verwende jetzt fuzzy matching
        if response:
            print(format_output(response))
            save_chat_to_txt(user_input, response, user_ip=user_ip, username=username)
        else:
            # Unbeantwortete Frage speichern und Fallback ausgeben
            print("Unanswered question detected. Saving to file...")
            save_unanswered_question(user_input, "data/unanswered_questions.json")
            response = fallback_responses["Fallback"]
            print(format_output(response))
            save_chat_to_txt(user_input, response, user_ip=user_ip, username=username)



if __name__ == "__main__":
    chat()
