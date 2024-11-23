from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from chatbot_main import get_faq_answer_fuzzy, save_chat_to_txt  # Importiere die benötigten Funktionen
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
import time
import json

# Lade Konfiguration aus config.json
def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Konfigurationsdatei nicht gefunden. Standardwerte werden verwendet.")
        return {
            "ip": "0.0.0.0",
            "port": 8000,
            "allow_origins": ["*"],
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    except json.JSONDecodeError as e:
        print(f"Fehler beim Laden der Konfigurationsdatei: {e}")
        return {
            "ip": "0.0.0.0",
            "port": 8000,
            "allow_origins": ["*"],
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

config = load_config()

# FastAPI-Anwendung erstellen
app = FastAPI()

# Statistik-Variablen
statistics = {
    "avg_response_time": 0,
    "total_requests": 0,
    "active_chats": 0,
    "start_time": time.time()
}

@app.get("/stats")
async def get_statistics():
    """Gibt Statistiken über die API-Nutzung zurück."""
    uptime = time.time() - statistics["start_time"]
    return {
        "uptime": uptime,
        "avg_response_time": statistics["avg_response_time"],
        "total_requests": statistics["total_requests"],
        "active_chats": statistics["active_chats"],
    }

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["allow_origins"],
    allow_credentials=True,
    allow_methods=config["allow_methods"],
    allow_headers=config["allow_headers"],
)

# Anfrage-Modell
class Query(BaseModel):
    question: str

# Funktion zur Anonymisierung der IP-Adresse
def anonymize_ip(ip_address):
    if ":" in ip_address:  # IPv6-Adresse
        parts = ip_address.split(":")
        if len(parts) > 2:
            parts[-1] = "xxxx"
            parts[-2] = "xxxx"
        return ":".join(parts)
    elif "." in ip_address:  # IPv4-Adresse
        parts = ip_address.split(".")
        if len(parts) == 4:
            parts[-1] = "xxx"  # Letztes Oktett anonymisieren
        return ".".join(parts)
    return ip_address

# Chatbot-Endpunkt
@app.post("/chat")
async def chat_with_bot(query: Query, request: Request):
    """Verarbeitet eine Anfrage an den Chatbot und gibt die Antwort zurück."""
    question = query.question
    if not question:
        raise HTTPException(status_code=400, detail="Frage fehlt in der Anfrage.")

    # IP-Adresse extrahieren und anonymisieren
    client_ip = request.client.host
    anonymized_ip = anonymize_ip(client_ip)

    # Chatbot-Logik anwenden
    response = get_faq_answer_fuzzy(question)

    # Chatverlauf speichern
    save_chat_to_txt(question, response, user_ip=anonymized_ip, username="Unbekannt")

    # Statistiken aktualisieren
    statistics["total_requests"] += 1
    return {"response": response}

# SSL-Konfiguration laden
def load_ssl_config():
    """Prüft, ob SSL-Zertifikate verfügbar sind, und gibt die Pfade zurück."""
    ssl_keyfile = "/home/ismail/Chatbot/SSL/privkey.pem"
    ssl_certfile = "/home/ismail/Chatbot/SSL/fullchain.pem"

    if os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile):
        return {"keyfile": ssl_keyfile, "certfile": ssl_certfile}
    return None

if __name__ == "__main__":
    ssl_config = load_ssl_config()

    # Host und Port aus config.json laden
    host = config.get("ip", "0.0.0.0")
    port = config.get("port", 8000)

    if ssl_config:
        print(f"SSL-Zertifikate gefunden. Server wird mit HTTPS gestartet auf https://{host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=ssl_config["keyfile"],
            ssl_certfile=ssl_config["certfile"]
        )
    else:
        print(f"SSL-Zertifikate nicht gefunden. Server wird mit HTTP gestartet auf http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
