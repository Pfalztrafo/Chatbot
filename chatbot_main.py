import json
import os
from datetime import datetime
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import sys

# Globale Variablen
config = None
faq_data = []
model = None
tokenizer = None
hf_pipeline = None

FALLBACK_ANSWER = "Entschuldigung, ich konnte keine passende Antwort finden. Bitte kontaktieren Sie unseren Support."

#Wechseln zwischen GPU und Server
if torch.cuda.is_available():
    device = 0
    #print("→ CUDA ist verfügbar, verwende GPU!")
else:
    device = None
    print("→ Keine GPU gefunden, verwende CPU.")
    
# -----------------------------------------
# Schritt 1: Config laden
def load_config():
    global config
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        # Validierung & Standardwerte
        print("[DEBUG] Konfiguration erfolgreich geladen.")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[FEHLER] Fehler beim Laden der config.json: {e}")
        sys.exit(1)

# -----------------------------------------
# Schritt 2: FAQ-Daten laden
def get_file_path(file_type):
    file_mapping = {
        "faq_general": "data/faq_general.json",
        "faq_sales": "data/faq_sales.json",
    }
    return file_mapping.get(file_type)

def load_faq_data():
    global faq_data
    faq_data = []
    for file_key in ["faq_general", "faq_sales"]:
        file_path = get_file_path(file_key)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                raw_data = json.load(file)
                # Überprüfen der Datenstruktur
                for entry in raw_data:
                    faq_data.append({
                        "question": entry.get("question", ""),
                        "answer": entry.get("answer", ""),
                        "synonyms": entry.get("synonyms", [])
                    })
        except FileNotFoundError:
            print(f"[ERROR] FAQ-Datei nicht gefunden: {file_path}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Fehler beim Lesen der FAQ-Datei: {e}")

    print(f"[DEBUG] {len(faq_data)} FAQs für Fuzzy Matching erfolgreich geladen.")

# -----------------------------------------
# Modell + Tokenizer laden
def load_model_and_tokenizer():
    global model, tokenizer, hf_pipeline
    MODEL_NAME = config["MODEL"]["MODEL_NAME"]
    MODEL_PATH = os.path.join(config["MODEL"]["MODEL_PATH"], MODEL_NAME.replace("/", "_"))
    
    if os.path.exists(MODEL_PATH):
        print(f"[DEBUG] Modellpfad existiert: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    else:
        print(f"[ERROR] Modellpfad nicht gefunden: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        device = 0
        #print("→ CUDA ist verfügbar, verwende GPU!")
    else:
        device = None
        print("→ Keine GPU gefunden, verwende CPU.")

    # Pipeline
    hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    print("[DEBUG] Pipeline erstellt.")

# -----------------------------------------
# FAQ-Funktion
def get_faq_answer(user_input, category=None, threshold=80):
    """ Findet die beste FAQ-Antwort basierend auf Fuzzy-Matching.
        Berücksichtigt sowohl `question` als auch `synonyms` in den FAQs.  """
    if not config["CHAT"].get("use_fuzzy_matching", True):
       #print("[DEBUG] Fuzzy-Matching ist deaktiviert.")
       return None, 0.0

    best_match = None
    best_score = 0
    for item in faq_data:
        # Fuzzy-Matching auf `question`
        question_score = fuzz.ratio(user_input.lower(), item["question"].lower()) / 100
        if question_score > best_score and question_score >= threshold / 100:
            best_match = item["answer"]
            best_score = question_score

        # Fuzzy-Matching auf `synonyms`
        for synonym in item.get("synonyms", []):
            synonym_score = fuzz.ratio(user_input.lower(), synonym.lower()) / 100
            if synonym_score > best_score and synonym_score >= threshold / 100:
                best_match = item["answer"]
                best_score = synonym_score

    return best_match, best_score

# Überprüfung der Score-Bereiche
def validate_scores(fuzzy_score):
    fuzzy_min, fuzzy_max = config["fuzzy_score_range"]
    if fuzzy_score < fuzzy_min or fuzzy_score > fuzzy_max:
        print("[DEBUG] Fuzzy-Score außerhalb des zulässigen Bereichs.")
        return False
    return True

# -----------------------------------------
def evaluate_response(reference, hypothesis):
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing_function)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(reference, hypothesis)
    return {
        "bleu": bleu_score,
        "rougeL": rouge_scores["rougeL"].fmeasure
    }

def get_faq_reference_for_evaluation(user_input, threshold=80):
    best_match = None
    best_score = 0

    for item in faq_data:
        question_score = fuzz.ratio(user_input.lower(), item["question"].lower()) / 100
        if question_score > best_score and question_score >= threshold/100:
            best_match = item["answer"]
            best_score = question_score

        for synonym in item.get("synonyms", []):
            synonym_score = fuzz.ratio(user_input.lower(), synonym.lower()) / 100
            if synonym_score > best_score and synonym_score >= threshold/100:
                best_match = item["answer"]
                best_score = synonym_score

    return best_match

# -----------------------------------------
# Chat speichern
def save_chat_to_txt(user_message, bot_response, evaluation=None, scores=None, generated_text=None, user_ip="user", folder="chat_logs", username=None):
    os.makedirs(folder, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(folder, f"{date_str}_chat_log.txt")

    # Standardwerte für Evaluation und Scores
    if evaluation is None:
        evaluation = {"bleu": 0.0, "rougeL": 0.0}
    if scores is None:
        scores = {}

    # Hole die relevanten Werte 
    fuzzy_score = scores.get("fuzzy", 0.0)
    log_score = scores.get("ki", -float("inf"))  # Fallback auf 0.0, falls nicht vorhanden

     # Fallback für log_score
    if log_score is None:
        log_score = -float("inf")  # Setze Standardwert für nicht vorhandene Log-Scores

    # Schreibe BLEU/ROUGE als Text
    eval_text = f"BLEU: {evaluation['bleu']:.2f}, ROUGE-L: {evaluation['rougeL']:.2f}"
    score_text = f"Fuzzy: {fuzzy_score:.2f}, Log-Score: {log_score:.2f}"

    # Schreibe in die Datei
    with open(filename, "a", encoding="utf-8") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] [IP: {user_ip}] {user_message}\n")
        #file.write(f"[{timestamp}] [Server] [Bot ({score_text})]: {bot_response} [{eval_text}]\n")
        file.write(f"[{timestamp}] Bot: {generated_text} [{eval_text}, {score_text}]\n")
        #if generated_text:
            #file.write(f"[{timestamp}] [Generated Text]: {generated_text}\n")

    # Debugging-Ausgabe im Terminal
    if fuzzy_score > 0.0:
        print(f"[DEBUG] FAQ-Matching: Fuzzy Score: {fuzzy_score:.2f}")
    #if log_score is not None:
        #print(f"[DEBUG] Log-Score: {log_score:.2f}")
    if generated_text:
        print(f"Bot: {generated_text}")
    pass

# -----------------------------------------
# Generative KI-Antwort
def generate_ki_response(query):    
    # Konfiguration für die interne Prompt-Nutzung
    internal_prompt = config["CHAT"].get("internal_prompt", "Beantworte Frage.")
    #input_text = f"{internal_prompt}\nFrage: {query}\nAntwort:"
    input_text = f"{internal_prompt}\nQuestion: {query}\nAnswer:"

    """ Kann entweder `hf_pipeline` oder den normalen Ansatz verwenden. """
    try:
        # 1) Referenzantwort für BLEU/ROUGE evaluieren
        # Finde die passende Referenzantwort aus den FAQs für Metriken
        best_faq_ref = get_faq_reference_for_evaluation(query, threshold=80)
        if not best_faq_ref:
            best_faq_ref = FALLBACK_ANSWER  # Falls keine Referenz gefunden, nutze Fallback

        # 2) Prüfen, ob hf_pipeline oder der normale Ansatz genutzt wird
        if config["CHAT"].get("use_pipeline", True):
            # Nutzung von hf_pipeline
            try:
                result = hf_pipeline(input_text, max_length=100, num_return_sequences=1)[0]
                #result = hf_pipeline(query, max_length=100, num_return_sequences=1)[0]
                generated_text = result["generated_text"]
                log_score = None  # hf_pipeline liefert keinen Log-Score
                if log_score is None:
                    log_score = -float("inf")  # Standardwert setzen

            except Exception as e:
                print(f"[ERROR] Fehler bei der Generierung mit hf_pipeline: {e}")
                return "", FALLBACK_ANSWER, -float("inf"), {"bleu": 0.0, "rougeL": 0.0}
        else:
            # Nutzung des normalen Ansatzes
            generate_kwargs = {
                "input_ids": tokenizer(input_text, return_tensors="pt").input_ids.to(device),
                "max_length": config["CHAT"]["max_length"],
                "min_length": config["CHAT"]["min_length"],
                "num_beams": config["CHAT"]["num_beams"],
                "repetition_penalty": config["CHAT"]["repetition_penalty"],
                "no_repeat_ngram_size": config["CHAT"]["no_repeat_ngram_size"],
                "length_penalty": config["CHAT"]["length_penalty"],
                "early_stopping": config["CHAT"]["early_stopping"],
                "return_dict_in_generate": True,
                "output_scores": True
            }

            # Optional: Sampling-Parameter hinzufügen, falls aktiviert
            if config["CHAT"].get("do_sample", False):
                generate_kwargs.update({
                    "do_sample": True,
                    "temperature": config["CHAT"]["temperature"],
                    "top_k": config["CHAT"]["top_k"],
                    "top_p": config["CHAT"]["top_p"]
                })

            outputs = model.generate(**generate_kwargs)
            generated_ids = outputs.sequences
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

            # Log-Score berechnen
            scores = outputs.scores
            probs = [torch.softmax(score, dim=-1) for score in scores]
            log_probs = [torch.log(prob) for prob in probs]
            log_score = sum(
                log_prob[0, token_id].item()
                for log_prob, token_id in zip(log_probs, generated_ids[0][1:])
            )

        # 3) BLEU & ROUGE evaluieren
        evaluation = evaluate_response(best_faq_ref, generated_text)

        # Debugging: Ausgabe von Bewertungen und generierter Antwort
        print(f"Log-Score: {log_score:.2f}, BLEU: {evaluation['bleu']:.2f}, ROUGE-L: {evaluation['rougeL']:.2f}")

        # 4) Fallback-Logik prüfen
        if log_score is not None and log_score < config["CHAT"]["log_score_threshold"]:
            return generated_text, FALLBACK_ANSWER, log_score, evaluation

        bleu_min, _ = config["CHAT"]["bleu_score_range"]
        rouge_min, _ = config["CHAT"]["rougeL_score_range"]

        if evaluation['bleu'] < bleu_min or evaluation['rougeL'] < rouge_min:
            return generated_text, FALLBACK_ANSWER, log_score, evaluation

        # Generierte Antwort verwenden
        return generated_text, generated_text, log_score, evaluation

    except Exception as e:
        print(f"[ERROR] Fehler bei der Generierung: {e}")
        return "", FALLBACK_ANSWER, -float("inf"), {"bleu": 0.0, "rougeL": 0.0}

# -----------------------------------------
# Kern: get_response

def get_response(user_input, user_ip="user"):
    # 1) FAQ
    response, fuzzy_score = get_faq_answer(user_input, threshold=config["CHAT"].get("fuzzy_threshold", 70))
    if response:
        save_chat_to_txt(user_input, response, None, {"fuzzy": fuzzy_score}, user_ip=user_ip)
        return response

    # 2) Generative KI
    if config["CHAT"].get("use_ki_generative", True):
        generated_text, final_response, log_score, evaluation = generate_ki_response(user_input)
        save_chat_to_txt(user_input, final_response, evaluation, {"fuzzy": 0.0, "ki": log_score}, generated_text, user_ip=user_ip)
        return final_response

    # 3) Fallback
    save_chat_to_txt(user_input, FALLBACK_ANSWER, user_ip=user_ip)
    return FALLBACK_ANSWER


# -----------------------------------------
# Interaktive Schleife
def chat():
    print("Starte den Chat (zum Beenden 'exit' eingeben)")
    while True:
        user_input = input("Du: ").strip()
        if user_input.lower() == "exit":
            print("Chat beendet.")
            break
        response = get_response(user_input)
        print(f"Bot: {response}")

# -----------------------------------------
# Initialisierung
def init_chatbot():
    global config, faq_data, model, tokenizer, hf_pipeline

    load_config()            # config laden
    load_faq_data()         # FAQ laden
    load_model_and_tokenizer()  # Modell & Tokenizer laden
    print("[DEBUG] init_chatbot() ist abgeschlossen.")


# -----------------------------------------
# Hauptfunktion
def main():
    init_chatbot()  # Alle nötigen Daten laden
    chat()          # Interaktiver Loop

# -----------------------------------------
if __name__ == "__main__":
    main()  # Starte das Programm
