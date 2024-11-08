import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
from utils import MODEL_NAME
from datetime import datetime
import os
import platform
import nltk
from nltk.corpus import wordnet as wn
import psutil  # Zum Abrufen der detaillierten Systeminformationen
import cpuinfo # CPU Infos holen

# WordNet-Daten einmalig herunterladen
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Optional für zusätzliche Sprachdaten

# LLM google/flan-t5-base laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Pfad zur Log-Datei
log_file_path = "./training_logs/training_logs.txt"

# Fortschritt-Datei für Trainingsepochen
progress_file = "training_logs/training_progress.json"

def load_progress():
    try:
        with open(progress_file, "r") as file:
            progress = json.load(file)
    except FileNotFoundError:
        progress = {"total_epochs": 0}
    return progress

def save_progress(progress):
    with open(progress_file, "w") as file:
        json.dump(progress, file)




# Systeminformationen und GPU-Verfügbarkeit
def get_device_spec():
    # Initialisiere die Gerätespezifikationen
    device_spec = {
        "Device": "GPU" if torch.cuda.is_available() else "CPU",
        "CPU": "Unknown CPU",
        "RAM": "Unknown RAM",
        "GPU": "Unknown GPU",
        "Platform": platform.platform()
    }
    
    # CPU-Informationen abrufen
    try:
        cpu_info = cpuinfo.get_cpu_info()
        device_spec["CPU"] = cpu_info.get("brand_raw", "Unknown CPU")
    except Exception:
        pass  # Falls ein Fehler auftritt, bleibt die CPU-Info auf "Unknown CPU"
    
    # RAM-Informationen in GB abrufen
    try:
        total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Umrechnung in GB
        device_spec["RAM"] = f"{total_memory:.2f} GB"
    except Exception:
        pass  # Falls ein Fehler auftritt, bleibt die RAM-Info auf "Unknown RAM"
    
    # GPU-Informationen abrufen, falls verfügbar
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)  # Umrechnung in GB
            device_spec["GPU"] = f"{gpu_name} ({gpu_memory} GB)"
        else:
            device_spec["GPU"] = "N/A"
    except Exception:
        pass  # Falls ein Fehler auftritt, bleibt die GPU-Info auf "Unknown GPU"
    
    return device_spec



# Trainingsdetails loggen mit verbesserter Struktur und Speicherinformationen
def log_training_details(training_args, total_epochs, device_spec, epoch_logs, total_training_time):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"--- Training Run ---\n")
        log_file.write(f"Start Time: {timestamp}\n")
        log_file.write(f"Total Epochs in This Run: {training_args.num_train_epochs}\n")
        log_file.write(f"Cumulative Total Epochs: {total_epochs}\n")
        log_file.write(f"Learning Rate: {training_args.learning_rate}\n")
        log_file.write(f"Batch Size per Device: {training_args.per_device_train_batch_size}\n")
        log_file.write(f"Weight Decay: {training_args.weight_decay}\n")

        # Gerätespezifikationen mit verbesserter Formatierung loggen
        log_file.write("Device Specifications:\n")
        log_file.write(f"  CPU: {device_spec.get('CPU', 'N/A')}\n")
        log_file.write(f"  RAM: {device_spec.get('RAM', 'N/A')}\n")
        log_file.write(f"  GPU: {device_spec.get('GPU', 'N/A')}\n")
        log_file.write(f"  Platform: {device_spec.get('Platform', 'N/A')}\n")

        # Epoch Details
        log_file.write("\nEpoch Details:\n")
        for epoch_log in epoch_logs:
            log_file.write(f"  Epoch {epoch_log['epoch_num']}: Loss = {epoch_log['loss']:.4f}, Training Time = {epoch_log['training_time']:.2f} seconds\n")

        # Durchschnittsverlust und Gesamttrainingszeit
        avg_loss = sum(log['loss'] for log in epoch_logs) / len(epoch_logs)
        log_file.write(f"\nAverage Loss for This Run: {avg_loss:.4f}\n")
        log_file.write(f"Total Training Time for This Run: {total_training_time:.2f} seconds\n")
        log_file.write("--- End of Training ---\n\n")



# Systeminformationen und GPU-Verfügbarkeit
def get_device_spec():
    device_spec = {}

    # CPU-Informationen abrufen
    processor_name = platform.processor() or platform.uname().processor or "Unknown CPU"
    device_spec["CPU"] = processor_name

    # RAM-Informationen in GB abrufen
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Umrechnung in GB
    device_spec["RAM"] = f"{total_memory:.2f} GB"

    # GPU-Informationen abrufen, falls vorhanden
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)  # in GB
        device_spec["GPU"] = f"{gpu_name} ({gpu_memory} GB)"
    else:
        device_spec["GPU"] = "N/A"

    # System- und Plattforminformationen hinzufügen
    device_spec["Platform"] = f"{platform.system()} {platform.platform()}"

    return device_spec



# Funktion zum Laden von OpenThesaurus-Synonymen
def load_openthesaurus_text(filepath="data/openthesaurus.txt"):
    synonyms_dict = {}
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            synonyms = line.strip().split(";")
            for word in synonyms:
                synonyms_dict[word] = synonyms
    return synonyms_dict

# Funktion, um englische Synonyme aus WordNet zu holen
def get_english_synonyms(word):
    synonyms = []
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

# Trainingsdaten laden und erweitern
def load_data(language="de"):
    data = []
    dialogues_file = f"data/dialogues_{language}.json"
    rules_file = f"data/decision_rules_{language}.json"

    with open(dialogues_file, "r", encoding="utf-8") as file:
        dialogues = json.load(file)
        for item in dialogues:
            data.append({"input": item["question"], "output": item["answer"]})

    with open(rules_file, "r", encoding="utf-8") as file:
        rules = json.load(file)
        for application, rule_data in rules.items():
            default_recommendation = rule_data.get("default_recommendation", "")
            if default_recommendation:
                data.append({"input": f"{application} Empfehlung", "output": default_recommendation})
            for condition in rule_data.get("conditions", []):
                param = condition["parameter"]
                threshold = condition["threshold"]
                data.append({
                    "input": f"{application} {param} > {threshold}",
                    "output": condition["recommendation_above"]
                })
                data.append({
                    "input": f"{application} {param} <= {threshold}",
                    "output": condition["recommendation_below"]
                })
    return data

# Daten um Synonyme erweitern
def expand_with_synonyms(data, german_synonyms_dict):
    expanded_data = []
    for item in data:
        input_text = item["input"]
        expanded_data.append(item)
        
        # Deutsche Synonyme hinzufügen
        german_synonyms = german_synonyms_dict.get(input_text, [])
        for synonym in german_synonyms:
            expanded_data.append({"input": synonym, "output": item["output"]})

        # Englische Synonyme hinzufügen
        english_synonyms = get_english_synonyms(input_text)
        for synonym in english_synonyms:
            expanded_data.append({"input": synonym, "output": item["output"]})
    
    return expanded_data

# Daten für das Modell vorbereiten
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Hauptfunktion für das Training in beiden Sprachen
def main():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    german_synonyms_dict = load_openthesaurus_text()
    languages = ["de", "en"]  # Unterstützte Sprachen

    all_training_data = []
    for lang in languages:
        training_data = load_data(language=lang)
        training_data = expand_with_synonyms(training_data, german_synonyms_dict)
        all_training_data.extend(training_data)

    dataset = Dataset.from_dict({
        "input": [item["input"] for item in all_training_data],
        "output": [item["output"] for item in all_training_data]
    })
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",   # Speicherort des Modells
        eval_strategy="no",                # Evaluation deaktiviert, nur Training
        learning_rate=2e-5,                # Feinabstimmungs-Lernrate
        per_device_train_batch_size=4,     # Batch-Größe pro Gerät (RTX 3050 6GB -> 4)
        num_train_epochs=1,                # Anzahl der Epochen für besseres Lernen
        weight_decay=0.01                  # Gewichtszerfall zur Vermeidung von Overfitting
    )

    # Fortschritt des Trainings laden und aktualisieren
    progress = load_progress()
    num_train_epochs = training_args.num_train_epochs
    progress["total_epochs"] += num_train_epochs
    save_progress(progress)

    device_spec = get_device_spec()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    epoch_logs = []
    start_time = datetime.now()
    for epoch in range(num_train_epochs):
        epoch_start_time = datetime.now()
        loss = trainer.train().training_loss
        epoch_training_time = (datetime.now() - epoch_start_time).total_seconds()
        
        # Speichere Informationen für diese Epoche
        epoch_logs.append({
            "epoch_num": epoch + 1,
            "loss": loss,
            "training_time": epoch_training_time
        })

    # Speichere das Modell und den Tokenizer am Ende des Trainings
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    
    total_training_time = (datetime.now() - start_time).total_seconds()
    log_training_details(training_args, progress["total_epochs"], device_spec, epoch_logs, total_training_time)

    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print("Das feingetunte Modell und der Tokenizer wurden erfolgreich gespeichert.")

if __name__ == "__main__":
    main()
