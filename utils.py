import torch

# Konfiguration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/flan-t5-base"

# Hilfsfunktionen
def preprocess_text(text):
    return text.strip().lower()

def format_output(response):
    return f"Bot: {response}"
