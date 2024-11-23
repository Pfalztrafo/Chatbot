from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from chatbot_main import get_faq_answer_fuzzy, save_chat_to_txt  # Importiere die benötigten Funktionen
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from fastapi import FastAPI
import time
import requests


app = FastAPI()

# Beispiel für globale Variablen zur Speicherung von Statistiken
statistics = {
    "avg_response_time": 0,
    "total_requests": 0,
    "active_chats": 0,
    "start_time": time.time()
}

@app.get("/stats")
async def get_statistics():
    # Durchschnittliche Antwortzeit simulieren (später ersetzen)
    uptime = time.time() - statistics["start_time"]
    avg_response_time = statistics["avg_response_time"]
    total_requests = statistics["total_requests"]
    active_chats = statistics["active_chats"]
    return {
        "uptime": uptime,
        "avg_response_time": avg_response_time,
        "total_requests": total_requests,
        "active_chats": active_chats
    }

# Konfiguration der CORS-Einstellungen
allow_origins = ["https://alphatrafo.de", "https://api.alphatrafo.de"]  # Erlaube spezifische Domains
allow_credentials = True
allow_methods = ["*"]  # Erlaube alle HTTP-Methoden
allow_headers = ["*"]  # Erlaube alle Header


def load_ssl_config():
    """Prüft, ob SSL-Zertifikate verfügbar sind und gibt die Pfade zurück."""
    ssl_keyfile = "/home/ismail/Chatbot/SSL/privkey.pem"
    ssl_certfile = "/home/ismail/Chatbot/SSL/fullchain.pem"

    if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        return {"keyfile": ssl_keyfile, "certfile": ssl_certfile}
    else:
        return None





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


if __name__ == "__main__":
    ssl_config = load_ssl_config()

    if ssl_config:
        print("SSL-Zertifikate gefunden. Server wird mit HTTPS gestartet.")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            ssl_keyfile=ssl_config["keyfile"],
            ssl_certfile=ssl_config["certfile"]
        )
    else:
        print("Keine SSL-Zertifikate gefunden. Server wird mit HTTP gestartet.")
        uvicorn.run(app, host="0.0.0.0", port=8000)
