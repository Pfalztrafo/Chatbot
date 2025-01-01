import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from datetime import datetime
import os
import platform
from nltk.corpus import wordnet as wn
import psutil  # Zum Abrufen der detaillierten Systeminformationen

import subprocess
import threading

# WordNet-Daten einmalig herunterladen
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Optional für zusätzliche Sprachdaten

# ---------------------------- TensorBoard ---------------------------------------------
def start_tensorboard(logdir="./fine_tuned_model", port=6006):
    """
    Startet TensorBoard in einem separaten Thread.
    :param logdir: Verzeichnis, in dem die TensorBoard-Logs gespeichert werden.
    :param port: Port, auf dem TensorBoard läuft.
    """
    def run_tensorboard():
        subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    threading.Thread(target=run_tensorboard, daemon=True).start()

# ---------------------------- Konfiguration ---------------------------------------------
# Konfigurationsparameter laden
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
            "weight_decay": 0.01
        }
    }

    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            return {**default_config, **config}  # Standardwerte ergänzen
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warnung: {e}. Standardkonfiguration wird verwendet.")
        return default_config


# Konfigurationsparameter laden
config = load_config()
MODEL_PATH = os.path.join(config["MODEL"]["MODEL_PATH"], config["MODEL"]["MODEL_NAME"].replace("/", "_"))
MODEL_NAME = config["MODEL"]["MODEL_NAME"]

# LLM google/flan-t5 laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


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
# -----------------------------------------------------------------------

# Trainingsdaten laden und erweitern
def load_data():
    """
    Dynamisches Laden von Trainingsdaten basierend auf `data_sources` in config.json.
    """
    data = []
    files_to_load = config["TRAINING"].get("data_sources", [])

    for file_path in files_to_load:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                dialogues = json.load(file)
                for item in dialogues:
                    data.append({
                        "input": item["question"],
                        "output": item["answer"],
                        "context": item.get("context", ""),  # Kontext optional
                        "category": item.get("category", "General")  # Kategorie berücksichtigen
                    })
        except FileNotFoundError:
            print(f"[WARNUNG] Datei {file_path} nicht gefunden. Überspringe diese Datei.")
        except json.JSONDecodeError as e:
            print(f"[FEHLER] Fehler beim Lesen von {file_path}: {e}. Überspringe diese Datei.")

    if not data:
        raise ValueError("Keine Trainingsdaten gefunden! Überprüfen Sie die JSON-Dateien.")
    return data


# Negativbeispiele filtern
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



# Daten für das Modell vorbereiten
def preprocess_function(examples):
    inputs = []
    max_context_length = 350  # Maximale Länge des Kontexts
    for category, question, context in zip(
        examples["category"], 
        examples["input"], 
        examples.get("context", [""] * len(examples["input"]))
    ):
        # Kontext kürzen, wenn zu lang
        if len(context) > max_context_length:
            context = context[:max_context_length]
        inputs.append(f"[{category}] {question} Kontext: {context}" if context else f"[{category}] {question}")

    targets = examples["output"]
    model_inputs = tokenizer(inputs, max_length=350, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=350, padding="max_length", truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs



# Synonyme für die Erweiterung der Trainingsdaten
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


# Hauptfunktion für das Training pro Sprache
def main():
    """
    Hauptfunktion für das Training des Modells.
    Lädt die Daten, führt das Training durch und speichert das Modell.
    """
    print("Starte das Training...")

    # Speicherort für das Modell
    #output_dir = "./fine_tuned_model"
    output_dir = MODEL_PATH
    checkpoint_path = os.path.join(output_dir, "trainer_state.json")

    # Fortschritt des Trainings laden und aktualisieren
    progress = load_progress(MODEL_NAME)
    total_epochs = progress.get("total_epochs", 0)

    # Checkpoints prüfen und laden
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0 and os.path.exists(checkpoint_path):
        print("Lade zuletzt gespeicherte Modellparameter...")
        model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
        resume_training = True  # Fortsetzung des Trainings
    else:
        print("Starte mit dem Basis-Modell...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        resume_training = False  # Neues Training

    # Trainingsdaten laden
    training_data = load_data()

    # Negativbeispiele filtern
    training_data = filter_negatives(training_data, negative_sample_rate=config["TRAINING"].get("negative_sample_rate", 0.5))

    # Synonyme in die Trainingsdaten erweitern (optional)
    training_data = expand_with_synonyms(training_data)

    # Dataset erstellen
    dataset = Dataset.from_dict({
        "input": [item["input"] for item in training_data],
        "output": [item["output"] for item in training_data],
        "category": [item["category"] for item in training_data]
    })
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Konfigurationsparameter laden
    #config = load_config()

    # Daten aufteilen (80% Training, 20% Validierung)
    train_ratio = config["TRAINING"]["train_ratio"]
    train_size = int(train_ratio * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    # Trainingsargumente festlegen
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # Evaluation nach jeder Epoche
        save_strategy="epoch",  # Modell nach jeder Epoche speichern
        save_total_limit=2,  # Nur die letzten 2 Checkpoints speichern
        learning_rate=config["TRAINING"]["learning_rate"],
        per_device_train_batch_size=config["TRAINING"]["batch_size"],
        num_train_epochs=config["TRAINING"]["epochs"],  # Anzahl der Epochen
        weight_decay=config["TRAINING"]["weight_decay"],
        #logging_dir="./training_logs",  # Logs für TensorBoard
        logging_steps=10,  # Log-Schritte
        #log_level="info",  # Zeige mehr Details im Terminal
        load_best_model_at_end=True,  # Beste Modellparameter am Ende laden
        #metric_for_best_model="eval_loss",  # Metrik zur Auswahl des besten Modells
        #greater_is_better=False  # Kleinere Verluste sind besser
    )

    # Fortschritt aktualisieren
    total_epochs += training_args.num_train_epochs
    save_progress(MODEL_NAME, {"total_epochs": total_epochs})

    # Gerätespezifikationen abrufen
    device_spec = get_device_spec()

    # Trainer einrichten
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Validierungsdaten
        tokenizer=tokenizer,  # Wichtig für Seq2Seq-Modelle
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Early Stopping aktivieren
    )

    # Training starten (mit Checkpoint-Fortsetzung)
    start_time = datetime.now()
    if resume_training:
        print("Setze Training vom letzten Checkpoint fort...")
        trainer.train(resume_from_checkpoint=output_dir)
    else:
        print("Starte neues Training...")
        trainer.train()
    
    # Gesamttrainingszeit berechnen
    training_time = (datetime.now() - start_time).total_seconds()

    # Modell und Tokenizer speichern
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Fortschritt aktualisieren
    total_epochs += training_args.num_train_epochs
    save_progress(MODEL_NAME, {"total_epochs": total_epochs})

    # Trainingsdetails loggen
    epoch_logs = []
    for i, log in enumerate(trainer.state.log_history):
        if "loss" in log:
            epoch_logs.append({
                "epoch_num": i + 1,
                "loss": log["loss"],
                "grad_norm": log.get("grad_norm", None),
                "learning_rate": log.get("learning_rate", 0.0)
            })

    log_training_details(
        training_args,
        total_epochs,
        get_device_spec(),
        epoch_logs,
        training_time,
        None  # Phase ist aktuell nicht vorhanden
    )

    print(f"Training abgeschlossen.")
    print(f"Das feingetunte Modell wurde unter {output_dir} gespeichert.")
    print(f"Total Training Time: {training_time:.2f} seconds\n")


if __name__ == "__main__":
    print("Starte TensorBoard...")
    start_tensorboard(logdir="./fine_tuned_model", port=6006)  # Logs-Verzeichnis und Port angeben
    print("TensorBoard läuft auf http://localhost:6006")
    main()