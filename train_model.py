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
import json


# WordNet-Daten einmalig herunterladen
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Optional für zusätzliche Sprachdaten



def load_config():
    """Lädt Konfigurationsparameter aus der Datei config.json."""
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            if not config:
                raise ValueError("Die Datei ist leer.")
            return config
    except FileNotFoundError:
        raise FileNotFoundError("Die Datei 'config.json' wurde nicht gefunden. Bitte erstellen.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Fehler beim Lesen der config.json: Ungültiges JSON-Format. {e}")



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
    try:
        with open(progress_file, "w") as file:
            json.dump(progress, file, indent=4)
    except Exception as e:
        print(f"Fehler beim Speichern des Fortschritts: {e}")



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


# Trainingsdaten laden und erweitern
def load_data():
    """
    Lädt Trainingsdaten aus mehreren FAQ-Dateien und berücksichtigt Kategorien.
    """
    data = []
    files_to_load = ["data/faq_general.json", "data/faq_sales.json"]

    for file_path in files_to_load:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                dialogues = json.load(file)
                for item in dialogues:
                    data.append({
                        "input": item["question"],
                        "output": item["answer"],
                        "category": item.get("category", "Allgemein")  # Kategorie berücksichtigen
                    })
        except FileNotFoundError:
            print(f"[WARNUNG] Datei {file_path} nicht gefunden. Überspringe diese Datei.")
        except json.JSONDecodeError as e:
            print(f"[FEHLER] Fehler beim Lesen von {file_path}: {e}. Überspringe diese Datei.")
    
    if not data:
        raise ValueError("Keine Trainingsdaten gefunden! Überprüfen Sie die JSON-Dateien.")
    return data



# Daten für das Modell vorbereiten
def preprocess_function(examples):
    inputs = [f"[{category}] {question}" for category, question in zip(examples["category"], examples["input"])]
    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs


# Hauptfunktion für das Training pro Sprache
def main():
    # Nur eine Sprache trainieren (z. B. Deutsch)
    print("Starte das Training...")

    # Speicherort für das Modell
    output_dir = "./fine_tuned_model"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Trainingsdaten laden
    training_data = load_data()

    # Dataset erstellen
    dataset = Dataset.from_dict({
        "input": [item["input"] for item in training_data],
        "output": [item["output"] for item in training_data],
        "category": [item["category"] for item in training_data]  # Kategorie hinzufügen
    })
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Konfigurationsparameter laden
    config = load_config()

    # Trainingsargumente festlegen
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=6,
        weight_decay=0.01
    )

    # Fortschritt des Trainings laden und aktualisieren
    progress = load_progress()
    num_train_epochs = training_args.num_train_epochs
    progress["total_epochs"] += num_train_epochs
    save_progress(progress)

    # Gerätespezifikationen abrufen
    device_spec = get_device_spec()

    # Trainer einrichten und Training starten
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

        # Informationen zu jeder Epoche speichern
        epoch_logs.append({
            "epoch_num": epoch + 1,
            "loss": loss,
            "training_time": epoch_training_time
        })

    # Modell und Tokenizer speichern
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    total_training_time = (datetime.now() - start_time).total_seconds()

    # Trainingsdetails loggen
    log_training_details(training_args, progress["total_epochs"], device_spec, epoch_logs, total_training_time)

    print(f"Training abgeschlossen.")
    print(f"Das feingetunte Modell wurde unter {output_dir} gespeichert.")
    print(f"Total Training Time: {total_training_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
