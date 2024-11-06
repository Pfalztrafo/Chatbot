import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
from utils import MODEL_NAME


# Fortschritt-Datei laden oder erstellen
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

# Bisherige Fortschritte laden
progress = load_progress()

# Anzahl der Epochen für diesen Lauf
num_train_epochs = 1
total_epochs = progress["total_epochs"] + num_train_epochs

# Fortschritte speichern
progress["total_epochs"] = total_epochs
save_progress(progress)

# Fortschrittsmeldung
print(f"Gesamte trainierte Epochen über alle Läufe: {progress['total_epochs']}")


# Modellname und Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Daten laden und vorbereiten
def load_data():
    data = []

    # Transformator-Informationen
    with open("data/trafo_info.json", "r", encoding="utf-8") as file:
        trafo_info = json.load(file)
        for item in trafo_info:
            data.append({"input": item["topic"], "output": item["info"]})

    # Dienstleistungen
    with open("data/pfalztrafo_services.json", "r", encoding="utf-8") as file:
        services = json.load(file)
        for item in services:
            data.append({"input": item["service"], "output": item["description"]})

    # FAQ
    with open("data/dialogues.json", "r", encoding="utf-8") as file:
        dialogues = json.load(file)
        for item in dialogues:
            data.append({"input": item["question"], "output": item["answer"]})

    # Empfehlungen
    with open("data/rules.json", "r", encoding="utf-8") as file:
        rules = json.load(file)
        for rule in rules:
            data.append({"input": rule["condition"], "output": rule["recommendation"]})

    return data

# Daten in Dataset umwandeln
training_data = load_data()
dataset = Dataset.from_dict({"input": [item["input"] for item in training_data], "output": [item["output"] for item in training_data]})

# Daten für das Modell vorbereiten
def preprocess_function(examples):
    inputs = examples["input"]
    targets = examples["output"]
    
    # Tokenisiere Eingaben und Ziele mit Padding auf eine maximale Länge
    model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True) # max 128 Token, truncation=True: Längere Sequenzen werden auf die maximale Länge abgeschnitten, um Speicherplatz zu sparen und das Modell zu entlasten.
    
    # Tokenisiere auch die Labels mit Padding
    labels = tokenizer(targets, max_length=128, padding="max_length", truncation=True).input_ids
    model_inputs["labels"] = labels
    
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Trainingskonfiguration
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",   # Speicherort des Modells
    eval_strategy="no",                # Evaluation deaktiviert, nur Training
    learning_rate=2e-5,                # Feinabstimmungs-Lernrate
    per_device_train_batch_size=1,     # Batch-Größe pro Gerät (RTX 3050 6GB -> 4)
    num_train_epochs=1,                # Anzahl der Epochen für besseres Lernen
    weight_decay=0.01                  # Gewichtszerfall zur Vermeidung von Overfitting
)

# Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Training starten
trainer.train()

# Modell speichern
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
