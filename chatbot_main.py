import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from modules.dialogue_manager import get_faq_answer, save_unanswered_question, get_related_faq
from modules.recommendation import get_advanced_recommendation
from utils import format_output, preprocess_text, MODEL_NAME
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import torch
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import single_meteor_score
from spellchecker import SpellChecker

# Globale Variablen
faq_data = None  # Initialisierung außerhalb der Funktion
config = None  # Konfiguration wird später geladen


# Dynamische Pfade für JSON-Dateien
def get_file_path(file_type):
    file_mapping = {
        "faq_general": "data/faq_general.json",
        "faq_sales": "data/faq_sales.json",
        "fallback_responses": "data/fallback_responses.json"
    }
    return file_mapping.get(file_type)

# Konfiguration laden
def load_config():
    global config
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Konfigurationsdatei nicht gefunden. Standardwerte werden verwendet.")
        config = {
            "use_fuzzy_matching": True,
            "use_embeddings": True,
            "use_ki_generative": True,
            "embedding_model": "sentence-transformers/all-mpnet-base-v2"
        }
    except json.JSONDecodeError as e:
        print(f"Fehler beim Laden der Konfigurationsdatei: {e}")
        config = {
            "use_fuzzy_matching": True,
            "use_embeddings": True,
            "use_ki_generative": True,
            "embedding_model": "sentence-transformers/all-mpnet-base-v2"
        }

# Modell und Tokenizer laden
def load_model_and_tokenizer():
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        print("Verwende das feingetunte Modell.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print("Kein feingetuntes Modell gefunden. Verwende das Standardmodell.")
    return model, tokenizer

# Wissensbasis für RAG vorbereiten
def setup_retriever(knowledge_base, embeddings):
    faiss_index = FAISS.from_documents(knowledge_base, embeddings)
    return faiss_index

# Wissensdatenbank laden
def load_knowledge_base():
    knowledge_base = []
    for file_key in ["faq_general", "faq_sales"]:
        file_path = get_file_path(file_key)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                raw_data = json.load(file)
            for item in raw_data:
                knowledge_base.append(Document(page_content=item["question"], metadata={"answer": item["answer"], "category": item.get("category", "Allgemein")}))
        except FileNotFoundError:
            print(f"[ERROR] FAQ-Datei nicht gefunden: {file_path}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Fehler beim Lesen der FAQ-Datei: {e}")
    print(f"[DEBUG] FAQ-Daten geladen: {len(knowledge_base)} Einträge")
    return knowledge_base

# Globale Initialisierung
MODEL_PATH = "./fine_tuned_model"
load_config()
device = 0 if torch.cuda.is_available() else -1
model, tokenizer = load_model_and_tokenizer()
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

embeddings = HuggingFaceEmbeddings(model_name=config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"))
knowledge_base = load_knowledge_base()
faiss_index = setup_retriever(knowledge_base, embeddings)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=faiss_index.as_retriever(), chain_type="stuff")

# Synonym- und Rechtschreibkorrektur
def normalize_and_correct_input(user_input, synonym_dict):
    """
    Normalisiert die Eingabe basierend auf OpenThesaurus-Synonymen und korrigiert Rechtschreibfehler.
    """
    spell = SpellChecker(language="de")  # PySpellChecker für Deutsch
    words = user_input.split()
    processed_words = []

    for word in words:
        # Synonym prüfen und ersetzen
        if word.lower() in synonym_dict:
            processed_words.append(synonym_dict[word.lower()])
        else:
            # Rechtschreibkorrektur, wenn kein Synonym gefunden wird
            corrected_word = spell.correction(word)
            processed_words.append(corrected_word if corrected_word else word)

    return " ".join(processed_words)

# OpenThesaurus-Daten laden
def load_openthesaurus_data(file_path):
    """
    Lädt die OpenThesaurus-Daten aus einer Textdatei und erstellt ein Synonym-Wörterbuch.
    """
    synonym_dict = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                synonyms = line.strip().split("|")
                for synonym in synonyms:
                    synonym_dict[synonym.lower()] = synonyms[0]  # Erster Begriff als Hauptbegriff
    except FileNotFoundError:
        print(f"[ERROR] OpenThesaurus-Datei nicht gefunden: {file_path}")
    return synonym_dict

# Globale Initialisierung für OpenThesaurus
synonym_dict = load_openthesaurus_data("data/openthesaurus.txt")  # Pfad anpassen



# Fuzzy und Embeddings: zentrale Logik
def get_best_faq_response(user_input):
    """
    Mehrstufige Verarbeitung von Benutzeranfragen:
    - Kategorieerkennung
    - Fuzzy Matching
    - Embedding-basierte Suche (Kategorie / Global)
    - Generative Antwort
    - Fallback
    """
    scores = {"fuzzy": 0.0, "embedding": 0.0, "ki": 0.0}
    response = None
    category_detected = detect_category(user_input)
    print(f"[DEBUG] Erkannte Kategorie: {category_detected}")

    # Konfigurationsoptionen
    use_fuzzy_matching = config.get("use_fuzzy_matching", True)
    use_embeddings = config.get("use_embeddings", True)
    use_ki_generative = config.get("use_ki_generative", True)

    fuzzy_threshold = config.get("fuzzy_threshold", 70)
    embedding_threshold = config.get("embedding_threshold", 0.7)
    ki_confidence_threshold = config.get("ki_confidence_threshold", 0.5)

    # Schritt 1: Fuzzy Matching (kategoriebasiert)
    if use_fuzzy_matching:
        print("[DEBUG] Direktes Fuzzy Matching aktiv...")
        fuzzy_response, fuzzy_score = get_faq_answer(user_input, threshold=fuzzy_threshold, category=category_detected)
        scores["fuzzy"] = fuzzy_score / 100
        if fuzzy_response:
            print(f"[DEBUG] Fuzzy Score: {scores['fuzzy']:.2f}")
            response = fuzzy_response
            return response, scores, category_detected

    # Schritt 2: Embedding-basierte Suche (kategoriebasiert)
    if use_embeddings:
        print("[DEBUG] Embeddings-basierte Suche innerhalb der Kategorie aktiv...")
        embedding_response, embedding_score, doc = search_faq_with_embeddings(user_input, category=category_detected, return_score=True)
        scores["embedding"] = embedding_score
        if embedding_response and embedding_score >= embedding_threshold:
            print(f"[DEBUG] Embedding Score (Kategorie): {scores['embedding']:.2f}")
            response = embedding_response
            category_detected = doc.metadata.get("category", "Allgemein")
            return response, scores, category_detected

    # Schritt 3: Embedding-basierte Suche (global, ohne Kategorie)
    if use_embeddings:
        print("[DEBUG] Embeddings-basierte Suche ohne Kategorie aktiv...")
        embedding_response, embedding_score, doc = search_faq_with_embeddings(user_input, category=None, return_score=True)
        scores["embedding"] = embedding_score

        # Nur Embedding-Antwort nutzen, wenn der Score den Schwellenwert erfüllt
        if embedding_response and embedding_score >= embedding_threshold:
            print(f"[DEBUG] Embedding Score (global): {scores['embedding']:.2f}")
            response = embedding_response
            category_detected = doc.metadata.get("category", "Allgemein")
            return response, scores, category_detected
        else:
            print(f"[DEBUG] Embedding Score ({embedding_score:.2f}) ist unter dem Schwellenwert ({embedding_threshold}). Generative KI wird verwendet...")



    # Schritt 4: Generative KI-Antwort
    if use_ki_generative:
        print("[DEBUG] Generative KI-Antwort aktiv...")
        response, ki_confidence = generate_ki_response(user_input)
        scores["ki"] = ki_confidence
        print(f"[DEBUG] KI Score: {scores['ki']:.2f}")
        if response and ki_confidence >= ki_confidence_threshold:
            category_detected = "Generative KI"
            return response, scores, category_detected

    # Schritt 5: Fallback
    print("[DEBUG] Keine passende Antwort gefunden. Fallback wird genutzt...")
    response = get_related_faq(user_input)
    save_unanswered_question(user_input)
    print(f"[DEBUG] Fallback-Antwort genutzt: {response}")

    return response, scores, category_detected



# Kategorie erkennen
def detect_category(user_input):
    """
    Erkennt die Kategorie basierend auf Schlüsselwörtern im Benutzerinput.
    """
    user_input = user_input.lower()

    # Kategorien definieren
    categories = {
        "Begrüßung": ["hi", "hallo", "hey", "guten tag", "servus", "grüß dich"],
        "Abschied": ["tschüss", "bis bald", "auf wiedersehen", "ciao", "mach's gut"],
        "Dank": ["danke", "vielen dank", "dankeschön", "danke dir", "danke schön"],
        "Systeminformation": ["wer bist du", "was kannst du", "wie funktioniert das", "bist du ein mensch"],
        "Zusätzliche Informationen": ["ich brauche hilfe", "hilfe bitte", "unterstützung", "ich benötige hilfe"],
        "Kauf": ["kaufen", "angebot", "preis", "bestellen", "verfügbarkeit"],
        "Lieferung": ["lieferung", "versand", "lieferzeit", "transport"],
        "Beratung": ["empfehlung", "entscheidung", "geeignet", "passend"],
        "Produktinformationen": ["informationen", "produkt", "details", "spezifikationen"],
        "Technik": ["spannung", "leistung", "anschluss", "technisch"],
        "Nachhaltigkeit & Qualität": ["nachhaltig", "energieeffizient", "qualität", "ökodesign"],
        "Vertrieb": ["verkaufen", "vertrieb", "angebot anfordern", "verkauf", "kundenservice"]
    }

    # Kategorie basierend auf Schlüsselwörtern erkennen
    for category, keywords in categories.items():
        if any(keyword in user_input for keyword in keywords):
            print(f"[DEBUG] Erkannte Kategorie: {category}")  # Debug-Ausgabe
            return category

    #print("[DEBUG] Erkannte Kategorie: Allgemein")  # Debug-Ausgabe
    return "Allgemein"




# Embeddings-Suche
def search_faq_with_embeddings(query, category=None, return_score=False):
    """
    Durchsucht die FAQs basierend auf Embeddings und berücksichtigt optional eine Kategorie.
    """
    embedding_vector = embeddings.embed_query(query)
    filtered_documents = knowledge_base

    # Filter nach Kategorie
    if category:
        filtered_documents = [doc for doc in knowledge_base if doc.metadata.get("category") == category]
        print(f"[DEBUG] Gefilterte Dokumente für Kategorie '{category}': {len(filtered_documents)}")

    # Fallback: Wenn keine Dokumente für die Kategorie gefunden werden
    if not filtered_documents:
        print("[DEBUG] Keine Dokumente für die Kategorie gefunden. Verwende gesamte Wissensbasis.")
        filtered_documents = knowledge_base

    # Suche mit Embeddings
    try:
        faiss_index = FAISS.from_documents(filtered_documents, embeddings)
        results = faiss_index.similarity_search_with_score_by_vector(embedding_vector, k=1)

        if results:
            best_result, raw_score = results[0]

            # FAISS-Rohscore direkt verwenden
            print(f"[DEBUG] FAISS-Roh-Score: {raw_score:.2f}")

            answer = best_result.metadata.get("answer", "")
            return (answer, raw_score, best_result) if return_score else answer
    except Exception as e:
        print(f"[DEBUG] Fehler bei der Embeddings-Suche: {e}")
    return (None, float('inf'), None) if return_score else None




# Generative KI-Antwort
def generate_ki_response(query):
    try:
        generated = hf_pipeline(query, max_length=100, num_return_sequences=1)[0]
        return generated["generated_text"], generated.get("score", 0.0)
    except Exception as e:
        print(f"[DEBUG] Fehler bei der KI-Generierung: {e}")
        return None, 0.0

# Berechnung von BLEU, ROUGE, METEOR
def calculate_bleu(reference, hypothesis):
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    return sentence_bleu(reference_tokens, hypothesis_tokens)

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure

def calculate_meteor(reference, hypothesis):
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    return single_meteor_score(reference_tokens, hypothesis_tokens)

# Chat-Logs speichern
def save_chat_to_txt(user_message, bot_response, scores, user_ip="Unbekannt", folder="chat_logs"):
    """
    Speichert den Chat-Verlauf in einer Textdatei mit Datum und Zeitstempel.
    """
    os.makedirs(folder, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(folder, f"{date_str}_chat_log.txt")
    with open(filename, "a", encoding="utf-8") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Scores mit zwei Nachkommastellen formatieren
        score_text = (f"Fuzzy: {scores['fuzzy']:.2f}, "
                      f"Emb.: {scores['embedding']:.2f}, "
                      f"KI: {scores['ki']:.2f}")
        file.write(f"[{timestamp}] [IP: {user_ip}] {user_message}\n")
        file.write(f"[{timestamp}] [Server] [Bot ({score_text})]: {bot_response}\n")

# Haupt-Chat-Funktion
def chat():
    print("Starte den Chat (zum Beenden 'exit' eingeben)")
    user_ip = "192.168.1.10"

    while True:
        user_input = input("Du: ").strip()
        if not user_input:
            print("[DEBUG] Leere Eingabe erkannt. Bitte geben Sie eine Frage ein.")
            continue
        if user_input.lower() == "exit":
            print("Chat beendet.")
            break

        response, scores, category_detected = get_best_faq_response(user_input)

        # Unbeantwortete Frage speichern
        if response == "Entschuldigung, ich konnte Ihre Anfrage nicht verstehen.":
            save_unanswered_question(user_input)

        print(format_output(response))
        save_chat_to_txt(user_input, response, scores, user_ip=user_ip)

if __name__ == "__main__":
    chat()
