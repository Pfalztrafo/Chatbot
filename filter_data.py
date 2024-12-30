import os
import json
import time
from sentence_transformers import SentenceTransformer, util

# Ordnerpfad
data_dir = "data/"

# Dateien, die gefiltert werden sollen
files_to_filter = [
    "GermanQuAD_train.json",
    "GermanDPR_train.json"
]

# Modell laden
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Referenztexte definieren
reference_texts = [
    # Technischer Support
    "Wie behebe ich einen Fehler in einem elektrischen Gerät?",
    "Was kann ich tun, wenn ein Gerät überhitzt?",
    "Welche Schritte sollte ich bei einer Fehlersuche beachten?",
    "Wie überprüfe ich die Spannung in einem Stromkreis?",
    "Wie erkenne ich einen Kurzschluss?",
    "Mein Gerät zeigt einen Fehlercode an. Was bedeutet das?",
    "Warum startet mein Gerät nicht?",
    "Was soll ich tun, wenn mein Gerät unerwartet neu startet?",
    "Wie kann ich meine Garantie in Anspruch nehmen?",
    
    # Reparatur und Energieversorgung
    "Was sind typische Probleme bei elektrischen Geräten?",
    "Wie tausche ich eine Sicherung aus?",
    "Wann sollte ich ein Gerät reparieren oder ersetzen?",
    "Wie finde ich heraus, ob ein Gerät zu viel Strom verbraucht?",
    "Was kann ich tun, wenn ein Stromausfall auftritt?",
    "Wie erkenne ich, ob ein Gerät überlastet ist?",
    "Welche Zeichen deuten darauf hin, dass ein Transformator defekt ist?",
    
    # Technik und Elektrotechnik
    "Was ist der Unterschied zwischen Stromstärke und Spannung?",
    "Wie funktioniert ein Stromkreis?",
    "Was ist der Unterschied zwischen Serienschaltung und Parallelschaltung?",
    "Wie messe ich den Widerstand eines Bauteils?",
    "Welche Funktion hat eine Sicherung in einem Stromkreis?",
    "Was ist ein Relais und wofür wird es verwendet?",
    "Wie unterscheidet sich ein Schalter von einem Sensor?",
    "Wie überprüfe ich, ob ein Schalter defekt ist?",
    
    # Netzwerke und Energie
    "Was ist der Unterschied zwischen Wechselstrom und Gleichstrom?",
    "Wie wird Strom in einem Haus verteilt?",
    "Welche Geräte benötigen eine Erdung?",
    "Wie erkenne ich, ob eine Erdung korrekt funktioniert?",
    "Was ist der Zweck eines Spannungswandlers?",
    "Warum schwankt die Spannung in meinem Hausnetz?",
    
    # Geräte und Anwendung
    "Welche Geräte benötigen einen Spannungswandler?",
    "Wie funktioniert ein Multimeter?",
    "Wie wähle ich die richtige Leistung für ein Gerät?",
    "Wie erkenne ich, ob ein Kabel defekt ist?",
    "Warum schaltet sich ein Gerät automatisch ab?",
    "Wie finde ich heraus, ob ein Akku defekt ist?",
    "Wie lange hält ein Akku normalerweise?",
    "Welche Batterien sind für mein Gerät geeignet?",
    "Wie erkenne ich ein kompatibles Ersatzteil für mein Gerät?",
    
    # Sicherheitsaspekte
    "Was sollte ich tun, bevor ich an einem Stromkreis arbeite?",
    "Welche Schutzausrüstung ist bei der Arbeit mit Elektrizität erforderlich?",
    "Was ist der Unterschied zwischen FI-Schutzschalter und Sicherung?",
    "Wie überprüfe ich, ob ein Gerät sicher ist?",
    "Welche Vorsichtsmaßnahmen gelten bei Hochspannung?",
    "Wie kann ich mich vor Stromschlägen schützen?",
    
    # Typische Fragen aus einer Kundenhotline
    "Mein Gerät funktioniert nicht mehr. Können Sie mir helfen?",
    "Wo finde ich das Handbuch für mein Gerät?",
    "Wie lange dauert die Reparatur meines Geräts?",
    "Kann ich Ersatzteile direkt bei Ihnen bestellen?",
    "Was kostet eine Reparatur?",
    "Mein Gerät macht seltsame Geräusche. Ist das normal?",
    "Wie aktualisiere ich die Software meines Geräts?",
    "Können Sie mir erklären, wie ich mein Gerät richtig einstelle?",
    "Warum dauert das Laden meines Akkus so lange?",
    "Wie kann ich meinen Stromverbrauch reduzieren?",
    "Was soll ich tun, wenn mein Gerät unangenehm riecht?",
    "Gibt es ein Austauschprogramm für ältere Geräte?"
]


reference_embeddings = model.encode(reference_texts, convert_to_tensor=True)

# Debugging-Funktion
def log_message(message):
    print(f"[DEBUG] {message}")

# Maximale Antwortlänge (in Zeichen)
MAX_ANSWER_LENGTH = 300

#SCHWELLENWERT
SCHWELLENWERT = 0.5

# Dateien filtern
for file_name in files_to_filter:
    start_time = time.time()  # Zeitmessung starten
    file_path = os.path.join(data_dir, file_name)

    if os.path.exists(file_path):
        log_message(f"Datei gefunden: {file_name}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filtered_data = []
        total_processed = 0
        relevant_found = 0
        entry_id = 1

        # Verarbeitung je nach Dateityp
        if isinstance(data, dict) and "data" in data:  # Für GermanQuAD
            log_message("Bearbeite GermanQuAD-Datei...")
            for entry in data["data"]:
                for paragraph in entry["paragraphs"]:
                    context = paragraph["context"]  # Kontext extrahieren
                    for qas in paragraph["qas"]:
                        question = qas["question"]
                        answer = qas["answers"][0]["text"] if qas["answers"] else "Keine Antwort"

                        # Antwortlängen-Filter
                        if len(answer) > MAX_ANSWER_LENGTH:
                            log_message(f"Antwort zu lang: {answer[:50]}...")
                            continue

                        embedding = model.encode(question, convert_to_tensor=True)
                        similarity = util.pytorch_cos_sim(reference_embeddings, embedding).max().item()

                        total_processed += 1
                        if similarity > SCHWELLENWERT:  # Schwellenwert anpassen
                            filtered_data.append({
                                "id": f"quad_{entry_id:05d}",
                                "question": question,
                                "answer": answer,
                                "context": context,  # Kontext hinzufügen
                                "category": "Training",  # Standardkategorie für GermanQuAD
                                "similarity": similarity
                            })
                            relevant_found += 1
                            entry_id += 1

                        # Debug: Fortschritt ausgeben
                        if total_processed % 10 == 0:
                            log_message(f"Bearbeitet: {total_processed} | Relevante gefunden: {relevant_found}")


        elif isinstance(data, list):  # Für GermanDPR
            log_message("Bearbeite GermanDPR-Datei...")
            for entry in data:
                question = entry["question"]
                answer = entry["answers"][0] if entry.get("answers") else "Keine Antwort"

                for ctx in entry.get("positive_ctxs", []):
                    context = ctx["text"]
                    embedding = model.encode(context, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(reference_embeddings, embedding).max().item()

                    # Antwortlängen-Filter
                    if len(answer) > MAX_ANSWER_LENGTH:
                        log_message(f"Antwort zu lang: {answer[:50]}...")
                        continue

                    total_processed += 1
                    if similarity > SCHWELLENWERT:
                        filtered_data.append({
                            "id": f"dpr_{entry_id:05d}",
                            "question": question,
                            "answer": answer,
                            "context": context,
                            "category": "Positive",  # Kategorie anstelle von type
                            "similarity": similarity
                        })
                        relevant_found += 1
                        entry_id += 1

                for ctx in entry.get("negative_ctxs", []):
                    context = ctx["text"]
                    embedding = model.encode(context, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(reference_embeddings, embedding).max().item()

                    total_processed += 1
                    if similarity > SCHWELLENWERT:
                        filtered_data.append({
                            "id": f"dpr_{entry_id:05d}",
                            "question": question,
                            "answer": "",  # Leere Antwort für Negativbeispiele
                            "context": context,
                            "category": "Negative",
                            "similarity": similarity
                        })
                        relevant_found += 1
                        entry_id += 1

                for ctx in entry.get("hard_negative_ctxs", []):
                    context = ctx["text"]
                    embedding = model.encode(context, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(reference_embeddings, embedding).max().item()

                    total_processed += 1
                    if similarity > SCHWELLENWERT:
                        filtered_data.append({
                            "id": f"dpr_{entry_id:05d}",
                            "question": question,
                            "answer": "",  # Leere Antwort für Hard-Negative
                            "context": context,
                            "category": "Hard-Negative",
                            "similarity": similarity
                        })
                        relevant_found += 1
                        entry_id += 1


       # Gefilterte Datei speichern
        filtered_file_name = file_name.replace(".json", "_filtered.json")
        filtered_file_path = os.path.join(data_dir, filtered_file_name)

        with open(filtered_file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)

        elapsed_time = time.time() - start_time
        log_message(f"Gefilterte Datei gespeichert: {filtered_file_path}")
        log_message(f"Verarbeitete Einträge: {total_processed}, Relevante gefunden: {relevant_found}")
        log_message(f"Bearbeitungszeit: {elapsed_time:.2f} Sekunden")
    else:
        log_message(f"Datei {file_name} wurde nicht im Ordner {data_dir} gefunden.")