from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from chatbot_main import get_faq_answer_fuzzy, save_chat_to_txt  # Importiere die benötigten Funktionen
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Konfiguration der CORS-Einstellungen
allow_origins = ["https://alphatrafo.de", "https://api.alphatrafo.de"]  # Erlaube spezifische Domains
allow_credentials = True
allow_methods = ["*"]  # Erlaube alle HTTP-Methoden
allow_headers = ["*"]  # Erlaube alle Header

# Funktion zur Anonymisierung der IP-Adresse
def anonymize_ip(ip_address):
    if ":" in ip_address:  # Prüfen, ob es sich um eine IPv6-Adresse handelt
        parts = ip_address.split(":")
        if len(parts) > 2:
            parts[-1] = "xxxx"
            parts[-2] = "xxxx"
        return ":".join(parts)
    elif "." in ip_address:  # Prüfen, ob es sich um eine IPv4-Adresse handelt
        parts = ip_address.split('.')
        if len(parts) == 4:
            parts[-1] = "xxx"  # Letztes Oktett anonymisieren
        return '.'.join(parts)
    return ip_address  # Gib die IP zurück, falls sie nicht anonymisiert werden kann

# Anfrage-Modell
class Query(BaseModel):
    question: str

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=allow_methods,
    allow_headers=allow_headers,
)

# Beispiel-Endpunkt für den Chatbot
@app.post("/chat")
async def chat_with_bot(query: Query, request: Request):
    # Frage aus der Anfrage abrufen
    question = query.question
    if not question:
        raise HTTPException(status_code=400, detail="Frage fehlt in der Anfrage.")

    # IP-Adresse aus der Anfrage extrahieren
    client_ip = request.client.host

    # Anonymisierte IP
    anonymized_ip = anonymize_ip(client_ip)

    # Chatbot-Logik anwenden (z. B. Fuzzy-Matching oder Embedding-Suche)
    response = get_faq_answer_fuzzy(question)

    # Chatverlauf speichern
    save_chat_to_txt(question, response, user_ip=anonymized_ip, username="Unbekannt")

    # Antwort zurückgeben
    return {"response": response}
