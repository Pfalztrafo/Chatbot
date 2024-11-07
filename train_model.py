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

# WordNet-Daten einmalig herunterladen
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Optional für zusätzliche Sprachdaten

# LLM google/flan-t5-base laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Pfad zur Log-Datei
log_file_path = "./training_logs/training_logs.txt"

# Fortschritt-Datei für Trainingsepochen
progress_file = "training_progress.json"

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

# Trainingsdetails loggen
def log_training_details(epoch, loss, total_epochs, training_args, training_time, device_spec):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"--- Training Run ---\n")
        log_file.write(f"Start Time: {timestamp}\n")
        log_file.write(f"Total Epochs: {total_epochs}\n")
        log_file.write(f"Learning Rate: {training_args.learning_rate}\n")
        log_file.write(f"Batch Size per Device: {training_args.per_device_train_batch_size}\n")
        log_file.write(f"Weight Decay: {training_args.weight_decay}\n")
        log_file.write(f"Device Specifications: {device_spec}\n")
        log_file.write(f"Training Epoch: {epoch+1}/{total_epochs}\n")
        log_file.write(f"Loss: {loss}\n")
        log_file.write(f"Training Time for Epoch: {training_time:.2f} seconds\n")
        log_file.write(f"--- End of Training ---\n\n")

# Systeminformationen und GPU-Verfügbarkeit
def get_device_spec():
    device_spec = {
        "Device": "GPU" if torch.cuda.is_available() else "CPU",
        "System": platform.system(),
        "Version": platform.version(),
        "Platform": platform.platform(),
        "Processor": platform.processor(),
        "RAM (GB)": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else "N/A"
    }
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
        output_dir="./fine_tuned_model",
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01
    )
    
    # Fortschritt des Trainings laden
    progress = load_progress()
    num_train_epochs = training_args.num_train_epochs
    total_epochs = progress["total_epochs"] + num_train_epochs
    progress["total_epochs"] = total_epochs
    save_progress(progress)

    device_spec = get_device_spec()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    start_time = datetime.now()
    for epoch in range(int(training_args.num_train_epochs)):
        epoch_start_time = datetime.now()
        loss = trainer.train().training_loss
        epoch_training_time = (datetime.now() - epoch_start_time).total_seconds()
        log_training_details(epoch, loss, total_epochs, training_args, epoch_training_time, device_spec)
    total_training_time = (datetime.now() - start_time).total_seconds()
    print(f"Total Training Time: {total_training_time:.2f} seconds")

if __name__ == "__main__":
    main()
