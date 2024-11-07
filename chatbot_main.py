from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from modules.knowledge_base import get_info_by_topic, get_service_description, load_faq_embeddings
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

# WordNet-Daten einmalig herunterladen
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Optional für zusätzliche Sprachdaten


# Variable für die aktuelle Spracheinstellung (Standard: Deutsch)
language = "de"

def set_language(user_input):
    global language
    if "switch to english" in user_input.lower():
        language = "en"
        print("Language switched to English.")
    elif "wechsel zu deutsch" in user_input.lower():
        language = "de"
        print("Sprache auf Deutsch umgestellt.")

# Dynamische Pfade für JSON-Dateien basierend auf der Sprache
def get_file_path(file_type):
    file_mapping = {
        "dialogues": f"data/dialogues_{language}.json",
        "decision_rules": f"data/decision_rules_{language}.json",
        "decision_trees": f"data/decision_trees_{language}.json",
        "fallback_responses": f"data/fallback_responses_{language}.json"
    }
    return file_mapping.get(file_type)


# Aktualisierte Kategorieerkennungsfunktion
def detect_category(user_input):
    user_input = user_input.lower()
    if any(keyword in user_input for keyword in ["wartung", "service", "reparatur"]):
        return "Service"
    elif any(keyword in user_input for keyword in ["kaufen", "empfehlen", "produkt", "preis"]):
        return "Kaufberatung"
    elif any(keyword in user_input for keyword in ["spannung", "leistung", "technisch", "transformator", "typ"]):
        return "Technik"
    else:
        return "Allgemein"




# Dynamisches Laden der Fallback-Antworten
def load_fallback_responses():
    with open(get_file_path("fallback_responses"), "r", encoding="utf-8") as file:
        return json.load(file)

# Fallback-Antwort basierend auf erkannter Kategorie
def fallback_response(user_input):
    responses = load_fallback_responses()
    category = detect_category(user_input)
    return responses.get(category, responses["Fallback"])

# Pfad zum trainierten Modell
model_path = "./fine_tuned_model"

# Überprüfen, ob das Modell existiert, und ein Standardmodell laden, falls es fehlt
if os.path.exists(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("Verwende das fine-tuned Modell.")
else:
    # Standardmodell von Hugging Face verwenden
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print("Verwende das Standardmodell von Hugging Face.")

# Gerät für die Pipeline festlegen (GPU falls verfügbar)
device = 0 if torch.cuda.is_available() else -1
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

#---------------------------------------------------------------

# Wissensdatenbank laden und in das passende Format konvertieren
def load_knowledge_base():
    with open(get_file_path("dialogues"), "r", encoding="utf-8") as file:
        raw_data = json.load(file)
    documents = [
        Document(page_content=item["question"], metadata={"answer": item["answer"]})
        for item in raw_data
    ]
    return documents

# Wissensbasis für RAG vorbereiten
def setup_retriever(knowledge_base):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    faiss_index = FAISS.from_documents(knowledge_base, embeddings)
    retriever = faiss_index.as_retriever()
    return retriever

# Initialisierung der Wissensdatenbank und des Retrievers
knowledge_base = load_knowledge_base()
retriever = setup_retriever(knowledge_base)

# RetrievalQA-Kette für die dynamische Antwortgenerierung
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Embeddings-Modell und FAISS-Index einmalig laden
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
faq_index = load_faq_embeddings()


# Fuzzy Matching und Embeddings-basierte Suche
def get_faq_answer_fuzzy(user_input):
    # Lade die ursprünglichen FAQ-Daten aus dialogues.json direkt
    faq_data = load_knowledge_base()  # Wir verwenden hier die Funktion load_knowledge_base, die die FAQ-Daten bereits lädt
    user_input = preprocess_text(user_input)

    question_variants = []
    question_to_answer = {}

    for item in faq_data:
        question_variants.append(item.page_content)
        question_to_answer[item.page_content] = item.metadata["answer"]

        # Prüfe auf Synonyme und füge diese zu den Variationen hinzu
        if "synonyms" in item.metadata:
            for synonym in item.metadata["synonyms"]:
                question_variants.append(synonym)
                question_to_answer[synonym] = item.metadata["answer"]

        # Englische Synonyme über WordNet hinzufügen
        english_synonyms = get_english_synonyms(item.page_content)
        for synonym in english_synonyms:
            question_variants.append(synonym)
            question_to_answer[synonym] = item.metadata["answer"]

        # Deutsche Synonyme über OpenThesaurus hinzufügen
        german_synonyms = german_synonyms_dict.get(item.page_content, [])
        for synonym in german_synonyms:
            question_variants.append(synonym)
            question_to_answer[synonym] = item.metadata["answer"]

    # Fuzzy Matching auf alle Fragevarianten anwenden
    best_match, score = process.extractOne(user_input, question_variants, scorer=fuzz.token_sort_ratio)

    # Wenn Fuzzy Matching eine hohe Übereinstimmung findet, wird diese Antwort zurückgegeben
    if score > 70:
        return question_to_answer[best_match]

    # Wenn Fuzzy Matching keine Übereinstimmung findet, Embeddings-basierte Suche als Fallback verwenden
    return search_faq_with_embeddings(user_input)



# Embeddings-basierte Suche als Fallback
def search_faq_with_embeddings(query):
    # Stelle sicher, dass query ein einzelner String ist
    embedding = embeddings_model.embed_query(query)  # Nutze das Embeddings-Modell zur Vektorisierung der Abfrage
    result = faq_index.similarity_search_by_vector(embedding, k=1)
    
    if result:
        return result[0].metadata["answer"]
    return "Ich habe leider keine passende Antwort gefunden."



# Chat-log
def save_chat_to_txt(user_message, bot_response, folder="chat_logs"):
    # Stelle sicher, dass der Ordner für die Chat-Logs existiert
    os.makedirs(folder, exist_ok=True)

    # Erstelle den Dateinamen basierend auf dem aktuellen Datum
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(folder, f"{date_str}_chat_log.txt")

    # Schreibe den Chatverlauf in die Datei
    with open(filename, "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] User: {user_message}\n")
        file.write(f"[{timestamp}] Bot: {bot_response}\n")


# Speichere unbefragte Fragen
def save_unanswered_question(user_message, filename="data/unanswered_questions.json"):
    question_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": user_message,
        "category": detect_category(user_message),
        "answer": ""
    }
    
    # Überprüfe, ob der Ordner "data" existiert, und erstelle ihn falls nötig
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        # Falls die Datei existiert, lade die Daten und überprüfe, ob es eine Liste ist
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
            # Falls data ein Dictionary ist, initialisiere als leere Liste
            if isinstance(data, dict):
                data = []
    except (FileNotFoundError, json.JSONDecodeError):
        # Falls die Datei noch nicht existiert oder leer ist, initialisiere als leere Liste
        data = []

    # Füge die neue unbeantwortete Frage zur Liste hinzu
    data.append(question_data)

    # Speichere die aktualisierten Daten zurück in die Datei
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# Funktion zum Laden der OpenThesaurus-Synonyme aus der Textdatei
def load_openthesaurus_text(filepath="data/openthesaurus.txt"):
    synonyms_dict = {}

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            # Jede Zeile enthält eine Gruppe von Synonymen, getrennt durch Semikolons
            synonyms = line.strip().split(";")
            for word in synonyms:
                # Ordne jedes Wort der gesamten Synonymgruppe zu
                synonyms_dict[word] = synonyms

    return synonyms_dict

# Synonyme laden und in eine globale Variable speichern
german_synonyms_dict = load_openthesaurus_text()

# Funktion, um englische Synonyme von WordNet abzurufen
def get_english_synonyms(word):
    synonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

# Haupt-Chat-Funktion
def chat():
    print("Starte den Chat (zum Beenden 'exit' eingeben)")
    while True:
        user_input = input("Du: ")
        if user_input.lower() == "exit":
            print("Chat beendet.")
            break

        # Prüfe und setze die Sprache basierend auf der Benutzereingabe
        set_language(user_input)
        user_input = preprocess_text(user_input)

        # Fortgeschrittene Empfehlung basierend auf Bedingungen in decision_rules.json
        if any(keyword in user_input for keyword in ["industrie", "privat"]):
            parameters = {
                "voltage": int(input("Geben Sie die Spannung ein (z.B. 10000 für 10 kV): ")),
                "kva": int(input("Geben Sie die KVA ein (z.B. 500): "))
            }
            response = get_advanced_recommendation(user_input, parameters)
        else:
            # FAQ mit Fuzzy Matching v1.0.0
            response = get_faq_answer_fuzzy(user_input) or qa_chain.run(user_input) or "Entschuldigung, dazu habe ich keine Informationen."
        
        # Speichere den Chatverlauf
        save_chat_to_txt(user_input, response)

        # Speichere unbeantwortete Fragen, falls keine Antwort gefunden wurde
        if response == "Ich habe leider keine Antwort auf diese Frage.":
            save_unanswered_question(user_input)

        print(format_output(response))

if __name__ == "__main__":
    chat()