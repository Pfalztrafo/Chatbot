import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from datetime import datetime
import os
import platform
import psutil  # Zum Abrufen der detaillierten Systeminformationen
import glob  # Zum Durchsuchen von Dateien
import subprocess
import threading

# ---------------------------- TensorBoard ---------------------------------------------
def start_tensorboard(logdir="./fine_tuned_model", port=6007):
    """
    Startet TensorBoard in einem separaten Thread.
    :param logdir: Verzeichnis, in dem die TensorBoard-Logs gespeichert werden.
    :param port: Port, auf dem TensorBoard läuft.
    """
    def run_tensorboard():
        subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    threading.Thread(target=run_tensorboard, daemon=True).start()


# ---------------------------- Konfiguration ---------------------------------------------
def load_config():
    """Lädt Konfigurationsparameter aus der Datei config.json oder verwendet Standardwerte."""
    default_config = {
        "MODEL": {
            "MODEL_PATH": "./fine_tuned_model",
            "MODEL_NAME": "google/flan-t5-small"
        },
        "TRAINING": {
            "epochs": 1,
            "learning_rate": 2e-5,
            "batch_size": 1,
            "weight_decay": 0.01,
            "train_ratio": 0.8,
            "negative_sample_rate": 0.5,
            "include_german": False,  # Standardwert für include_german
            "training_args": {
                "eval_strategy": "epoch",
                "save_strategy": "epoch",
                "save_total_limit": 5,
                "logging_steps": 10,
                "load_best_model_at_end": True,
                "early_stopping_patience": 3
            }
        }
    }

    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            # Standardwerte ergänzen und zurückgeben
            final_config = {**default_config, **config}

            return final_config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warnung: {e}. Standardkonfiguration wird verwendet.")
        return default_config


config = load_config()  # Global definiert
tokenizer = None  # Platzhalter für den Tokenizer

# ---------------------------- Loggen ---------------------------------------------
# Pfad zur Log-Datei
log_file_path = "./training_logs/training_logs.txt"

# Fortschritt-Datei für Trainingsepochen
progress_file = "training_logs/training_progress.json"

def load_progress(model_name):
    try:
        with open(progress_file, "r") as file:
            progress = json.load(file)
        return progress.get(model_name, {"total_epochs": 0})
    except FileNotFoundError:
        return {"total_epochs": 0}

def save_progress(model_name, progress):
    try:
        with open(progress_file, "r") as file:
            all_progress = json.load(file)
    except FileNotFoundError:
        all_progress = {}

    all_progress[model_name] = progress
    with open(progress_file, "w") as file:
        json.dump(all_progress, file, indent=4)

# Trainingsdetails loggen mit verbesserter Struktur und Speicherinformationen
def log_training_details(training_args, total_epochs, device_spec, epoch_logs, total_training_time, phase=None):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"--- Training Run ---\n")
        log_file.write(f"Start Time: {timestamp}\n")
        if phase:
            log_file.write(f"Training Phase: {phase}\n")  # Phase ins Log schreiben
        log_file.write(f"Total Epochs in This Run: {training_args.num_train_epochs}\n")
        log_file.write(f"Cumulative Total Epochs: {total_epochs}\n")
        log_file.write(f"Learning Rate: {training_args.learning_rate}\n")
        log_file.write(f"Batch Size per Device: {training_args.per_device_train_batch_size}\n")
        log_file.write(f"Weight Decay: {training_args.weight_decay}\n")

        log_file.write("Device Specifications:\n")
        log_file.write(f"  CPU: {device_spec.get('CPU', 'N/A')}\n")
        log_file.write(f"  RAM: {device_spec.get('RAM', 'N/A')}\n")
        log_file.write(f"  GPU: {device_spec.get('GPU', 'N/A')}\n")
        log_file.write(f"  Platform: {device_spec.get('Platform', 'N/A')}\n")

        log_file.write("\nEpoch Details:\n")
        for epoch_log in epoch_logs:
            log_file.write(f"  Epoch {epoch_log['epoch_num']}: Loss = {epoch_log['loss']:.4f}, Grad Norm = {epoch_log['grad_norm']}, LR = {epoch_log['learning_rate']}\n")

        # Durchschnittsverlust berechnen
        if epoch_logs:
            avg_loss = sum(log['loss'] for log in epoch_logs) / len(epoch_logs)
            log_file.write(f"\nAverage Loss for This Run: {avg_loss:.4f}\n")
        else:
            log_file.write("\nAverage Loss for This Run: N/A (No epochs completed)\n")

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

# ---------------------------- Trainingsdaten laden und erweitern ---------------------------------------------

def load_faq_data():
    data = []
    faq_files = ["data/faq_sales.json", "data/faq_general.json"]

    for file_path in faq_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    for item in file_data:
                        data.append({
                            "input": item["question"],
                            "output": item["answer"],
                            "category": item.get("category", "Allgemein")  # Kategorie berücksichtigen
                        })
            except json.JSONDecodeError as e:
                print(f"[ERROR] Fehler beim Lesen von {file_path}: {e}")
        else:
            print(f"[WARNUNG] Datei {file_path} nicht gefunden. Überspringe diese Datei.")

    if not data:
        raise ValueError("Keine FAQ-Daten gefunden! Überprüfen Sie die JSON-Dateien.")
    return data


def load_german_data():
    training_data = []
    
    # GermanQuAD
    with open("data/GermanQuAD_train_filtered.json", "r", encoding="utf-8") as f:
        quad_data = json.load(f)
        for item in quad_data:
            training_data.append({
                "input": item["question"],    # Umbenennen von 'question' zu 'input'
                "output": item["answer"],     # Umbenennen von 'answer' zu 'output'
                "context": item["context"],
                "category": "Training"  # Standardkategorie für GermanQuAD
            })
    
    # GermanDPR
    with open("data/GermanDPR_train_filtered.json", "r", encoding="utf-8") as f:
        dpr_data = json.load(f)
        for item in dpr_data:
            training_data.append({
                "input": item["question"],    # Umbenennen von 'question' zu 'input'
                "output": item["answer"],     # Umbenennen von 'answer' zu 'output'
                "context": item["context"],
                "category": item.get("type", "Unknown")  # Nutze den Typ von DPR (Positive, Negative, etc.)
            })

    return training_data




# Daten für das Modell vorbereiten
def preprocess_faq_data(examples):
    inputs = [f"[{category}] {question}" for category, question in zip(examples["category"], examples["input"])]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs

def preprocess_german_data(examples):
    # Kombiniere Kontext und Eingabe (Frage) immer
    inputs = [f"{context} {input_text}" for context, input_text in zip(examples["context"], examples["input"])]
    targets = examples["output"]
    # Tokenisierung der Eingaben
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    # Tokenisierung der Labels (Antworten)
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True).input_ids
    # Setze die Labels
    model_inputs["labels"] = labels
    return model_inputs



# ---------------------------- Synonymerweiterung ---------------------------------------------
def expand_with_synonyms(data):
    expanded_data = []
    for item in data:
        expanded_data.append(item)  # Originalfrage
        if "synonyms" in item:
            for synonym in item["synonyms"]:
                expanded_data.append({
                    "question": synonym,
                    "answer": item["answer"],
                    "category": item["category"]
                })
    return expanded_data

# ---------------------------- Negativbeispiele filtern ---------------------------------------------
def filter_negatives(data, negative_sample_rate=0.5):
    """
    Filtert Negative und Hard-Negative Beispiele basierend auf der Sampling-Rate.
    """
    filtered_data = []
    for item in data:
        if item["category"] in ["Negative", "Hard-Negative"]:
            if torch.rand(1).item() <= negative_sample_rate:  # Zufällige Auswahl
                filtered_data.append(item)
        else:
            filtered_data.append(item)  # Positive und Training bleiben unverändert
    return filtered_data


# ---------------------------- Hauptfunktion für das Training ---------------------------------------------
def main():
    print("Starte das Training...")

    # Boolean aus der Konfiguration, um den Modus zu bestimmen
    include_german = config["TRAINING"].get("include_german", False)

    # Modell- und Ausgabeverzeichnis festlegen
    output_dir = os.path.abspath(config["MODEL"]["dynamic_model_paths"].get(config["MODEL"]["MODEL_NAME"], config["MODEL"]["MODEL_PATH"]))
    MODEL_NAME = config["MODEL"]["MODEL_NAME"]

    # Initialisiere den Tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Checkpoints prüfen und laden
    checkpoint_dir = os.path.join(output_dir, "checkpoint-*")
    checkpoints = sorted(glob.glob(checkpoint_dir))  # Alle Checkpoints suchen und sortieren
    checkpoint_path = checkpoints[-1] if checkpoints else None  # Letzten Checkpoint verwenden

    if checkpoint_path:
        print(f"Setze Training vom letzten Checkpoint ({checkpoint_path}) fort...")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        resume_training = True  # Fortsetzung des Trainings
    else:
        print("Starte mit dem Basis-Modell...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        resume_training = False  # Neues Training


    # Trainingsdaten laden
    # German-Daten laden
    if include_german:
        print("Lade German-Daten (GermanDPR und GermanQuAD)...")
        training_data = load_german_data()
        
        # Dataset erstellen
        dataset = Dataset.from_dict({
            "input": [item["input"] for item in training_data],
            "output": [item["output"] for item in training_data],
            "context": [item["context"] for item in training_data],
            "category": [item.get("category", "Allgemein") for item in training_data]
        })
        tokenized_dataset = dataset.map(preprocess_german_data, batched=True)
        
        # Filter Negative Beispiele
        negative_sample_rate = config["TRAINING"].get("negative_sample_rate", 0.5)
        training_data = filter_negatives(training_data, negative_sample_rate)

    # FAQ-Daten laden
    else:
        print("Lade FAQ-Daten (faq_sales und faq_general)...")
        training_data = load_faq_data()
        training_data = expand_with_synonyms(training_data)  # Synonyme in die Trainingsdaten erweitern
        
        # Dataset erstellen
        dataset = Dataset.from_dict({
            "input": [item["input"] for item in training_data],
            "output": [item["output"] for item in training_data],
            "category": [item["category"] for item in training_data]
        })
        tokenized_dataset = dataset.map(preprocess_faq_data, batched=True)



    # Daten aufteilen (80% Training, 20% Validierung)
    split_datasets = tokenized_dataset.train_test_split(test_size=1 - config["TRAINING"]["train_ratio"])
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # Trainingsargumente festlegen
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["TRAINING"]["epochs"],
        learning_rate=config["TRAINING"]["learning_rate"],
        per_device_train_batch_size=config["TRAINING"]["batch_size"],
        weight_decay=config["TRAINING"]["weight_decay"],
        evaluation_strategy=config["TRAINING"]["training_args"]["eval_strategy"],
        save_strategy=config["TRAINING"]["training_args"]["save_strategy"],
        save_total_limit=config["TRAINING"]["training_args"]["save_total_limit"],
        logging_steps=config["TRAINING"]["training_args"]["logging_steps"],
        load_best_model_at_end=config["TRAINING"]["training_args"]["load_best_model_at_end"]
        #logging_dir="./fine_tuned_model/logs",
        #save_safetensors=True  # Aktiviert die Safetensors-Unterstützung
    )

    # Fortschritt des Trainings laden
    progress = load_progress(MODEL_NAME)
    total_epochs = progress.get("total_epochs", 0)

    # Trainer einrichten
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["TRAINING"]["training_args"].get("early_stopping_patience", 3))
        ]
    )

    # Startzeit für die Gesamttrainingszeit
    start_time = datetime.now()

    # Training starten (mit Checkpoint-Fortsetzung)
    if resume_training:
        print(f"Setze Training vom letzten Checkpoint ({checkpoint_path}) fort...")
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        print("Starte neues Training...")
        trainer.train()

    # Modell und Tokenizer speichern
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    #------------------------------------------------------------------------
    # Gesamttrainingszeit berechnen
    training_time = (datetime.now() - start_time).total_seconds()

    # Fortschritt aktualisieren
    progress["total_epochs"] += training_args.num_train_epochs
    save_progress(MODEL_NAME, progress)

    # Trainingsdetails loggen
    epoch_logs = []
    for log in trainer.state.log_history:
        if "loss" in log:
            epoch_logs.append({
                "epoch_num": trainer.state.epoch,
                "loss": log["loss"],
                "grad_norm": log.get("grad_norm", None),
                "learning_rate": log.get("learning_rate", 0.0)
            })

    device_spec = get_device_spec()
    log_training_details(training_args, total_epochs, device_spec, epoch_logs, training_time)

    print(f"Training abgeschlossen.")
    print(f"Das feingetunte Modell wurde unter {output_dir} gespeichert.")
    print(f"Total Training Time: {training_time:.2f} seconds\n")

# ---------------------------- Skript starten ---------------------------------------------
if __name__ == "__main__":
    print("Starte TensorBoard...")
    start_tensorboard(logdir="./fine_tuned_model", port=6007)  # Logs-Verzeichnis und Port angeben
    print("TensorBoard läuft auf http://localhost:6007")
    main()