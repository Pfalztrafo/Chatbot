import json
import os
from datetime import datetime
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import torch
from nltk.translate.bleu_score import SmoothingFunction
import sys

# Globale Variablen
faq_data = []
config = None

# Konfiguration laden und validieren
def load_config():
    """
    Lädt die Konfigurationsdatei und setzt Standardwerte dynamisch, falls notwendig.
    Validiert wichtige Parameter und gibt Debugging-Informationen aus.
    """
    global config
    try:
        # Laden der Konfigurationsdatei
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Validierung von MODEL_NAME
        if config["MODEL"]["MODEL_NAME"] not in config["MODEL"].get("available_models", []):
            raise ValueError(f"Ungültiges MODEL_NAME: {config['MODEL']['MODEL_NAME']}")

        # Dynamische Standardwertsetzung und Grenzwerte validieren
        config["CHAT"]["use_fuzzy_matching"] = bool(config["CHAT"].get("use_fuzzy_matching", False))
        config["CHAT"]["use_ki_generative"] = bool(config["CHAT"].get("use_ki_generative", True))
        config["CHAT"]["use_pipeline"] = bool(config["CHAT"].get("use_pipeline", True))
        config["CHAT"]["fuzzy_threshold"] = max(0, min(100, config["CHAT"].get("fuzzy_threshold", 70)))
        config["CHAT"]["fuzzy_score_range"] = config["CHAT"].get("fuzzy_score_range", [0.5, 1.0])
        config["CHAT"]["bleu_score_range"] = config["CHAT"].get("bleu_score_range", [0.3, 0.5])
        config["CHAT"]["rougeL_score_range"] = config["CHAT"].get("rougeL_score_range", [0.4, 0.6])
        config["CHAT"]["do_sample"] = bool(config["CHAT"].get("do_sample", False))
        config["CHAT"]["temperature"] = max(0.0, min(1.0, config["CHAT"].get("temperature", 0.7)))
        config["CHAT"]["top_k"] = max(1, config["CHAT"].get("top_k", 50))
        config["CHAT"]["top_p"] = max(0.0, min(1.0, config["CHAT"].get("top_p", 0.9)))
        config["CHAT"]["num_beams"] = max(1, config["CHAT"].get("num_beams", 5))
        config["CHAT"]["repetition_penalty"] = max(1.0, config["CHAT"].get("repetition_penalty", 1.2))
        config["CHAT"]["max_length"] = max(1, config["CHAT"].get("max_length", 50))
        config["CHAT"]["min_length"] = max(0, min(config["CHAT"]["max_length"], config["CHAT"].get("min_length", 10)))
        config["CHAT"]["no_repeat_ngram_size"] = max(0, config["CHAT"].get("no_repeat_ngram_size", 3))
        config["CHAT"]["length_penalty"] = max(0.0, config["CHAT"].get("length_penalty", 1.0))
        config["CHAT"]["early_stopping"] = bool(config["CHAT"].get("early_stopping", True))
        config["CHAT"]["internal_prompt"] = config["CHAT"].get(
            "internal_prompt", "Bitte beantworte Fragen"
        )

        # Debugging wichtiger Parameter
        print(f"[DEBUG] Konfiguration erfolgreich geladen.")
        print(f"[DEBUG] Wichtige Parameter: {', '.join([f'{key}: {value}' for key, value in config['CHAT'].items()])}")
        if not config["CHAT"].get("use_fuzzy_matching", True):
            print("[DEBUG] Fuzzy-Matching ist deaktiviert.")

    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
        # Beenden des Programms bei Fehlern in der Konfiguration
        print(f"[FEHLER] Fehler beim Laden der config.json: {e}")
        sys.exit(1)  # Programm beenden
        
# ----------------------------------------------------------------------------------
# FAQs laden für Fuzzy Matching
# Dynamische Pfade für JSON-Dateien - Fuzyy Matching
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


# Antwort auf eine FAQ-Frage basierend auf Fuzzy-Matching
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

#------------------------------------------------------------------------------------
# Modell laden
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

# Konfiguration laden
load_config()  # Lädt die Konfiguration aus config.json
MODEL_NAME = config["MODEL"]["MODEL_NAME"]
MODEL_PATH = os.path.join(config["MODEL"]["MODEL_PATH"], MODEL_NAME.replace("/", "_"))

if os.path.exists(MODEL_PATH):
    print(f"[DEBUG] Modellpfad existiert: {MODEL_PATH}")
else:
    print(f"[ERROR] Modellpfad nicht gefunden: {MODEL_PATH}")

load_faq_data()
device = 0 if torch.cuda.is_available() else -1
model, tokenizer = load_model_and_tokenizer()
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

#------------------------------------------------------------------------------------
# KI
# Referenzantwort für BLEU/ROUGE evaluieren
def get_faq_reference_for_evaluation(user_input, threshold=80):
    best_match = None
    best_score = 0

    for item in faq_data:
        # Falls du denselben Fuzzy-Ansatz willst:
        question_score = fuzz.ratio(user_input.lower(), item["question"].lower()) / 100
        if question_score > best_score and question_score >= threshold/100:
            best_match = item["answer"]
            best_score = question_score

        for synonym in item.get("synonyms", []):
            synonym_score = fuzz.ratio(user_input.lower(), synonym.lower()) / 100
            if synonym_score > best_score and synonym_score >= threshold/100:
                best_match = item["answer"]
                best_score = synonym_score

    return best_match  # Kann None sein, wenn nichts passt

# Globale Fallback-Antwort
FALLBACK_ANSWER = "Entschuldigung, ich konnte keine passende Antwort finden. Bitte kontaktieren Sie unseren Support."


# Generative KI-Antwort mit Log-Score und Bewertung
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

# Bewertung von BLEU und RougeL
def evaluate_response(reference, hypothesis):
    smoothing_function = SmoothingFunction().method1  # Glättungsfunktion
    bleu_score = sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothing_function)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(reference, hypothesis)
    return {
        "bleu": bleu_score,
        "rougeL": rouge_scores["rougeL"].fmeasure
    }

#------------------------------------------------------------------------------------
# Chat-Logs speichern
def save_chat_to_txt(user_message, bot_response, evaluation=None, scores=None, generated_text=None, user_ip="user", folder="chat_logs", username=None):
    os.makedirs(folder, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(folder, f"{date_str}_chat_log.txt")

    # Standardwerte für Evaluation und Scores
    if evaluation is None:
        evaluation = {"bleu": 0.0, "rougeL": 0.0}
    if scores is None:
        scores = {}

    # Hole die relevanten Werte – wenn nichts da, nimm 0.0
    fuzzy_score = scores.get("fuzzy", 0.0)
    log_score = scores.get("ki", -float("inf"))  # Fallback auf 0.0, falls nicht vorhanden

     # Fallback für log_score, falls None
    if log_score is None:
        log_score = -float("inf")  # Setze Standardwert für nicht vorhandene Log-Scores

    # Schreibe BLEU/ROUGE als Text
    eval_text = f"BLEU: {evaluation['bleu']:.2f}, ROUGE-L: {evaluation['rougeL']:.2f}"
    score_text = f"Fuzzy: {fuzzy_score:.2f}, Log-Score: {log_score:.2f}"

    # Schreibe in die Datei
    with open(filename, "a", encoding="utf-8") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"[{timestamp}] [IP: {user_ip}] {user_message}\n")
        file.write(f"[{timestamp}] [Server] [Bot ({score_text})]: {bot_response} [{eval_text}]\n")
        if generated_text:
            file.write(f"[{timestamp}] [Generated Text]: {generated_text}\n")

    # Debugging-Ausgabe im Terminal
    if fuzzy_score > 0.0:
        print(f"[DEBUG] FAQ-Matching: Fuzzy Score: {fuzzy_score:.2f}")
    #if log_score is not None:
        #print(f"[DEBUG] Log-Score: {log_score:.2f}")
    if generated_text:
        print(f"Bot: {generated_text}")

#------------------------------------------------------------------------------------
# Hauptantwortlogik
def get_response(user_input):
    # Schritt 1: Suche in den FAQs
    response, fuzzy_score = get_faq_answer(user_input, threshold=config["CHAT"]["fuzzy_threshold"])
    if response:
        save_chat_to_txt(user_input, response, None, {"fuzzy": fuzzy_score})
        return response

    # Schritt 2: Generative KI
    if config["CHAT"]["use_ki_generative"]:
        # Vier Werte von generate_ki_response zurückgeben
        generated_text, final_response, log_score, evaluation = generate_ki_response(user_input)
        
        # Speichern Sie sowohl die generierte als auch die finale Antwort
        save_chat_to_txt(user_input, final_response, evaluation, {"fuzzy": 0.0, "ki": log_score}, generated_text)
        return final_response

    # Schritt 3: Fallback
    save_chat_to_txt(user_input, FALLBACK_ANSWER, {"bleu": 0.0, "rougeL": 0.0}, {"fuzzy": 0.0, "ki": 0.0})
    return FALLBACK_ANSWER


# Haupt-Chat-Funktion
def chat():
    print("Starte den Chat (zum Beenden 'exit' eingeben)")
    while True:
        user_input = input("Du: ").strip()
        if user_input.lower() == "exit":
            print("Chat beendet.")
            break
        response = get_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
