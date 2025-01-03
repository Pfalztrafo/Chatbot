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

# ---------------------------- Checkpoints ---------------------------------------------
def get_last_checkpoint(output_dir):
    """
    Findet den letzten Checkpoint im Ausgabe-Verzeichnis, der eine gültige Modelldatei enthält.
    """
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    valid_checkpoints = [
        ckpt for ckpt in checkpoints
        if os.path.exists(os.path.join(ckpt, "pytorch_model.bin")) or
           os.path.exists(os.path.join(ckpt, "model.safetensors"))
    ]
    if valid_checkpoints:
        # Sortiere die Checkpoints nach Nummer und wähle den neuesten
        checkpoints_sorted = sorted(valid_checkpoints, key=lambda x: int(x.split("-")[-1]))
        return checkpoints_sorted[-1]
    return None



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
            "data_sources": [
                "data/faq_general.json"
            ],
            "data_total": [
                "data/faq_general.json",
                "data/faq_sales.json",
                "data/GermanDPR_train_filtered.json",
                "data/GermanQuAD_train_filtered.json"
            ],
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
            return {**default_config, **config}  # Standardwerte ergänzen
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
def load_data(include_german=False):
    """
    Lädt Trainingsdaten aus JSON-Dateien. Kann wahlweise alle Daten (inkl. GermanDPR/QuAD) laden.
    :param include_german: Boolean, ob GermanDPR und GermanQuAD-Daten einbezogen werden sollen.
    """
    data = []
    files_to_load = config["TRAINING"]["data_sources"] if not include_german else config["TRAINING"]["data_total"]

    for file_path in files_to_load:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                entries = json.load(file)
                for item in entries:
                    data.append({
                        "input": item.get("question", "").strip(),
                        "output": item.get("answer", "").strip(),
                        "category": item.get("category", "General"),
                        "context": item.get("context", ""),  # Kontext wird nur für German genutzt
                        "synonyms": item.get("synonyms", [])
                    })
        except FileNotFoundError:
            print(f"[WARNUNG] Datei nicht gefunden: {file_path}")
        except json.JSONDecodeError as e:
            print(f"[FEHLER] Fehler beim Lesen von {file_path}: {e}")
    
    if not data:
        raise ValueError("Keine gültigen Trainingsdaten gefunden!")
    
    # Synonymerweiterung nur für FAQs
    if not include_german:
        data = expand_with_synonyms(data)
    return data


def expand_with_synonyms(data):
    """
    Erweitert Trainingsdaten mit Synonymen, falls vorhanden.
    """
    expanded_data = []
    for item in data:
        expanded_data.append(item)  # Originalfrage
        for synonym in item.get("synonyms", []):
            expanded_data.append({
                "input": synonym,
                "output": item["output"],
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

# ---------------------------- Daten für das Modell vorbereiten ---------------------------------------------
def preprocess_function(examples, use_context):
    """
    Bereitet die Eingaben für das Modell vor. Kann optional Kontext und Kategorien einbeziehen.
    :param use_context: Boolean, um kontextbasierte Verarbeitung zu aktivieren.
    """
    inputs = []
    targets = []

    for category, question, context, answer in zip(
        examples["category"], 
        examples["input"], 
        examples.get("context", [""] * len(examples["input"])),  # Kontext optional
        examples["output"]
    ):
        if use_context and context:
            # Kontextbasierte Eingaben
            input_text = f"Kategorie: {category}\nKontext: {context}\nFrage: {question}"
        else:
            # Einfache Eingaben für FAQs
            input_text = f"[{category}] {question}"

        inputs.append(input_text)
        targets.append(answer)

    model_inputs = tokenizer(inputs, max_length=350, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=350, padding="max_length", truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs


# ---------------------------- Hauptfunktion für das Training ---------------------------------------------
def main():
    print("Starte das Training...")

    # Boolean aus der Konfiguration, um den Modus zu bestimmen
    include_german = config["TRAINING"].get("include_german", False)

    output_dir = os.path.abspath(os.path.join(config["MODEL"]["MODEL_PATH"], config["MODEL"]["MODEL_NAME"].replace("/", "_")))
    MODEL_NAME = config["MODEL"]["MODEL_NAME"]

    global tokenizer  # Damit die Variable global verfügbar ist
    last_checkpoint = get_last_checkpoint(output_dir)

    try:
        if last_checkpoint:
            print(f"Inhalt des Checkpoints: {os.listdir(last_checkpoint)}")
            print(f"Checkpoint gefunden: {last_checkpoint}")
            model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint, ignore_mismatched_sizes=True)
            tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
            resume_training = True
        else:
            print("Kein gültiger Checkpoint gefunden. Starte mit dem Basis-Modell.")
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            resume_training = False
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}. Starte mit dem Basis-Modell.")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        resume_training = False

    # Debugging für Modell-Keys
    model_state_dict = model.state_dict()
    required_keys = ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight']
    missing_keys = set(required_keys) - set(model_state_dict.keys())
    print(f"Fehlende Keys: {missing_keys}")
    print(f"Geladene Parameter im Modell: {list(model_state_dict.keys())[:10]}")  # Zeige die ersten 10 Parameter


    # Fortschritt des Trainings laden
    progress = load_progress(MODEL_NAME)
    total_epochs = progress.get("total_epochs", 0)

    # Trainingsdaten laden und optional mit Kontext verarbeiten
    training_data = load_data(include_german=include_german)

    # Dataset erstellen
    dataset = Dataset.from_dict({
        "input": [item["input"] for item in training_data],
        "output": [item["output"] for item in training_data],
        "category": [item["category"] for item in training_data],
        "context": [item.get("context", "") for item in training_data]  # Kontext optional
    })

    # Boolean für Kontextverarbeitung
    use_context = include_german

    tokenized_dataset = dataset.map(lambda examples: preprocess_function(examples, use_context=use_context), batched=True)

    # Daten aufteilen (80% Training, 20% Validierung)
    split_datasets = tokenized_dataset.train_test_split(test_size=1 - config["TRAINING"]["train_ratio"])
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # Trainingsargumente festlegen
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=config["TRAINING"]["training_args"]["eval_strategy"],
        save_strategy=config["TRAINING"]["training_args"]["save_strategy"],
        save_total_limit=config["TRAINING"]["training_args"]["save_total_limit"],
        learning_rate=config["TRAINING"]["learning_rate"],
        per_device_train_batch_size=config["TRAINING"]["batch_size"],
        num_train_epochs=config["TRAINING"]["epochs"],
        weight_decay=config["TRAINING"]["weight_decay"],
        #logging_dir="./fine_tuned_model/logs",
        logging_steps=config["TRAINING"]["training_args"]["logging_steps"],
        load_best_model_at_end=config["TRAINING"]["training_args"]["load_best_model_at_end"],
        save_safetensors=True  # Aktiviert die Safetensors-Unterstützung
    )


    # Trainer einrichten
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Ab v5.0
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["TRAINING"]["training_args"].get("early_stopping_patience", 3))
        ]
    )


    # Training starten
    start_time = datetime.now()
    if resume_training:
        print(f"Setze Training vom letzten Checkpoint ({last_checkpoint}) fort...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Starte neues Training...")
        trainer.train()

    # Gesamttrainingszeit berechnen
    training_time = (datetime.now() - start_time).total_seconds()

    # Modell und Tokenizer speichern
    model.save_pretrained(output_dir, safe_serialization=False)

    #model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

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