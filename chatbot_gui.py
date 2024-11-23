import tkinter as tk
from tkinter import ttk, messagebox
from train_model import main as start_training_process  # Importiere die main-Funktion aus train_model.py
from api_main import allow_origins, allow_credentials, allow_methods, allow_headers, app  # Importiere die FastAPI-App und Parameter
from chatbot_main import get_faq_answer_fuzzy  # Importiere die benötigten Funktionen
from utils import preprocess_text
import os
import threading
import uvicorn
import json
from tqdm import tqdm
import psutil
import requests




class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Management Dashboard")
        self.root.geometry("800x600")

         # Konfiguration laden
        self.config = self.load_config()

        #Statistik
        self.update_statistics()


        # Globale Variable für Logs definieren
        self.chat_logs_dir = "chat_logs"
        os.makedirs(self.chat_logs_dir, exist_ok=True)

        # API-Status
        self.api_status = "Offline"

        # API starten
        self.start_api_server()

        # Widgets erstellen
        self.create_widgets()

        # Systemstatistiken starten (nach dem Erstellen der Widgets)
        self.update_system_stats()

    def start_api_server(self):
        """Startet den FastAPI-Server in einem separaten Thread."""
        def run_server():
            try:
                uvicorn.run(app, host="0.0.0.0", port=8000)
            except Exception as e:
                print(f"Fehler beim Starten der API: {e}")
                self.api_status = "Offline"

        # Server in einem separaten Thread starten
        api_thread = threading.Thread(target=run_server, daemon=True)
        api_thread.start()
        self.api_status = "Online"

    
    def update_system_stats(self):
        """Aktualisiert die CPU- und RAM-Auslastung in der Übersicht."""
        # CPU-Auslastung abrufen
        cpu_usage = psutil.cpu_percent(interval=0.1)
        self.cpu_label.config(text=f"CPU: {cpu_usage}%")

        # RAM-Auslastung abrufen
        ram = psutil.virtual_memory()
        total_ram = ram.total / (1024**3)  # Gesamt-RAM in GB
        used_ram = ram.used / (1024**3)   # Verwendeter RAM in GB
        self.ram_label.config(text=f"RAM: {used_ram:.1f} GB von {total_ram:.1f} GB")

        # Wiederholtes Aktualisieren der Werte
        self.root.after(1000, self.update_system_stats)  # Alle 1 Sekunde aktualisieren


    def create_widgets(self):
        # Tabs erstellen
        tab_control = ttk.Notebook(self.root)

        self.overview_tab = ttk.Frame(tab_control)
        self.training_tab = ttk.Frame(tab_control)
        self.logs_tab = ttk.Frame(tab_control)
        self.test_tab = ttk.Frame(tab_control)
        self.settings_tab = ttk.Frame(tab_control)

        tab_control.add(self.overview_tab, text="Übersicht")
        tab_control.add(self.training_tab, text="Training")
        tab_control.add(self.logs_tab, text="Logs")
        tab_control.add(self.test_tab, text="Testen")
        tab_control.add(self.settings_tab, text="Einstellung")
        tab_control.pack(expand=1, fill="both")

        # Inhalte der Tabs erstellen
        self.create_overview_tab()
        self.create_training_tab()
        self.create_logs_tab()
        self.create_test_tab()
        self.create_settings_tab()


#------------------------
    def load_config(self):
        """Lädt Konfigurationsparameter aus config.json."""
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "ip": "127.0.0.1",
                "port": 8000,
                "allow_origins": ["*"],
                "allow_methods": ["*"],
                "allow_headers": ["*"],
                "epochs": 1,
                "learning_rate": 0.00002,
                "batch_size": 4
            }
        except json.JSONDecodeError as e:
            messagebox.showerror("Fehler", f"Ungültiges JSON-Format: {e}")
            self.root.quit()

    def save_config(self):
        """Speichert die aktuellen Konfigurationsparameter in config.json."""
        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("Erfolg", "Einstellungen gespeichert.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern der Konfiguration: {e}")



    # Tab: Übersicht
    def create_overview_tab(self):
        ttk.Label(self.overview_tab, text="CPU- und RAM-Auslastung", font=("Arial", 16)).pack(pady=10)
        self.cpu_label = ttk.Label(self.overview_tab, text="CPU: 0%", font=("Arial", 12))
        self.cpu_label.pack(pady=5)
        self.ram_label = ttk.Label(self.overview_tab, text="RAM: 0 GB von 0 GB", font=("Arial", 12))
        self.ram_label.pack(pady=5)

        # Nach Erstellung der Labels die Systemstatistiken starten
        ttk.Label(self.overview_tab, text="Statistiken", font=("Arial", 16)).pack(pady=10)
        self.response_time_label = ttk.Label(self.overview_tab, text="Durchschnittliche Antwortzeit: 0 ms", font=("Arial", 12))
        self.response_time_label.pack(pady=5)
        self.total_requests_label = ttk.Label(self.overview_tab, text="Gesamtanzahl der Anfragen: 0", font=("Arial", 12))
        self.total_requests_label.pack(pady=5)
        self.active_chats_label = ttk.Label(self.overview_tab, text="Aktive Chats: 0", font=("Arial", 12))
        self.active_chats_label.pack(pady=5)

        ttk.Label(self.overview_tab, text="API-Status", font=("Arial", 16)).pack(pady=10)
        self.api_status_label = ttk.Label(self.overview_tab, text=f"API: {self.api_status}", foreground="green" if self.api_status == "Online" else "red", font=("Arial", 12))
        self.api_status_label.pack(pady=5)

        ttk.Label(self.overview_tab, text="Version", font=("Arial", 16)).pack(pady=10)
        self.version_label = ttk.Label(self.overview_tab, text="Version: 1.2.0", font=("Arial", 12))
        self.version_label.pack(pady=5)

        # SSL-Status anzeigen
        ssl_status = "Aktiviert" if os.path.exists("/home/ismail/chatbot/SSL/privkey.pem") and os.path.exists("/home/ismail/chatbot/SSL/fullchain.pem") else "Deaktiviert"
        self.ssl_status_label = ttk.Label(self.overview_tab, text=f"SSL-Status: {ssl_status}", font=("Arial", 12))
        self.ssl_status_label.pack(pady=5)

        self.quit_button = ttk.Button(self.overview_tab, text="Beenden", command=self.root.quit)
        self.quit_button.pack(pady=10)

        # Nach Erstellung der Labels die Statistiken und Systemauslastung starten
        self.update_system_stats()
        self.update_statistics()


    def update_statistics(self):
        """Holt echte Statistiken von der API und aktualisiert die Labels."""
        try:
            # API-Endpunkt für Statistiken
            response = requests.get("http://127.0.0.1:8000/stats")
            if response.status_code == 200:
                stats = response.json()
                avg_response_time = stats.get("avg_response_time", 0)
                total_requests = stats.get("total_requests", 0)
                active_chats = stats.get("active_chats", 0)

                # Labels aktualisieren
                self.response_time_label.config(text=f"Durchschnittliche Antwortzeit: {avg_response_time} ms")
                self.total_requests_label.config(text=f"Gesamtanzahl der Anfragen: {total_requests}")
                self.active_chats_label.config(text=f"Aktive Chats: {active_chats}")
            else:
                raise ValueError("Fehler beim Abrufen der Statistiken von der API.")
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Statistiken: {e}")

        # Wiederholtes Aktualisieren der Werte
        self.root.after(30000, self.update_statistics)  # Aktualisierung alle 5 Sekunden

    # Tab: Training
    def create_training_tab(self):
        """Erstellt den Training-Tab."""
        ttk.Label(self.training_tab, text="Training Parameter", font=("Arial", 16)).pack(pady=10)

        # Eingabefelder für Trainingsparameter
        ttk.Label(self.training_tab, text="Epochen:").pack(pady=5)
        self.epochs_entry = ttk.Entry(self.training_tab)
        self.epochs_entry.insert(0, self.config["epochs"])  # Initialwert aus config.json
        self.epochs_entry.pack(pady=5)

        ttk.Label(self.training_tab, text="Lernrate:").pack(pady=5)
        self.lr_entry = ttk.Entry(self.training_tab)
        self.lr_entry.insert(0, self.config["learning_rate"])  # Initialwert aus config.json
        self.lr_entry.pack(pady=5)

        ttk.Label(self.training_tab, text="Batchgröße:").pack(pady=5)
        self.batch_entry = ttk.Entry(self.training_tab)
        self.batch_entry.insert(0, self.config["batch_size"])  # Initialwert aus config.json
        self.batch_entry.pack(pady=5)

        # Fortschrittsanzeige
        ttk.Label(self.training_tab, text="Trainingsfortschritt:", font=("Arial", 12)).pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.training_tab, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

        # Trainingslogs
        ttk.Label(self.training_tab, text="Trainingslogs:", font=("Arial", 12)).pack(pady=10)
        self.training_logs_text = tk.Text(self.training_tab, height=10, width=80, state="disabled")
        self.training_logs_text.pack(pady=5)

        # Trainingsbutton
        self.train_button = ttk.Button(self.training_tab, text="Training starten", command=self.start_training)
        self.train_button.pack(pady=10)



    # Tab: Logs
    def create_logs_tab(self):
        ttk.Label(self.logs_tab, text="Chat Logs", font=("Arial", 16)).pack(pady=10)
        self.logs_text = tk.Text(self.logs_tab, height=20, width=80)
        self.logs_text.pack(pady=10)
        self.display_logs()

    def display_logs(self):
        log_files = sorted(
            [f for f in os.listdir(self.chat_logs_dir) if f.endswith(".txt")],
            key=lambda x: os.path.getctime(os.path.join(self.chat_logs_dir, x)),
            reverse=True
        )
        if log_files:
            log_path = os.path.join(self.chat_logs_dir, log_files[0])
            try:
                with open(log_path, "r", encoding="utf-8") as file:
                    logs = file.read()
            except UnicodeDecodeError:
                with open(log_path, "r", encoding="latin-1") as file:
                    logs = file.read()
            self.logs_text.insert(tk.END, logs)
        else:
            self.logs_text.insert(tk.END, "Keine Logs verfügbar.")

    # Tab: Testen
    def create_test_tab(self):
        ttk.Label(self.test_tab, text="Chat Terminal", font=("Arial", 16)).pack(pady=10)
        self.chat_history = tk.Text(self.test_tab, height=20, width=80, state="disabled")
        self.chat_history.pack(pady=10)

        self.test_input = ttk.Entry(self.test_tab, width=70)
        self.test_input.pack(pady=10)
        self.test_input.bind("<Return>", self.handle_test_input)

    def handle_test_input(self, event):
        user_input = self.test_input.get()
        if not user_input:
            return
        self.test_input.delete(0, tk.END)
        self.append_to_chat("Du", user_input)

        response = get_faq_answer_fuzzy(preprocess_text(user_input))
        self.append_to_chat("Bot", response)

    def append_to_chat(self, sender, message):
        self.chat_history.config(state="normal")
        self.chat_history.insert(tk.END, f"{sender}: {message}\n")
        self.chat_history.config(state="disabled")
        self.chat_history.see(tk.END)


    def apply_settings(self):
        """Speichert die geänderten API-Einstellungen in config.json."""
        try:
            # Werte aus den Eingabefeldern lesen
            self.config["ip"] = self.ip_entry.get()
            self.config["port"] = int(self.port_entry.get())
            self.config["allow_origins"] = self.origins_entry.get().split(", ")
            self.config["allow_methods"] = self.methods_entry.get().split(", ")
            self.config["allow_headers"] = self.headers_entry.get().split(", ")

            # Änderungen in der Konfigurationsdatei speichern
            self.save_config()

            # Erfolgsmeldung
            messagebox.showinfo("Erfolg", "Einstellungen wurden erfolgreich gespeichert.")
        except ValueError:
            # Fehler abfangen, falls ungültige Eingaben gemacht wurden
            messagebox.showerror("Fehler", "Ungültiger Wert für Port oder andere Felder.")
        except Exception as e:
            # Allgemeine Fehlerbehandlung
            messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten: {e}")




    # Tab: Einstellung
    def create_settings_tab(self):
        """Erstellt den Einstellungen-Tab."""
        ttk.Label(self.settings_tab, text="API Einstellungen", font=("Arial", 16)).pack(pady=10)

        # IP-Adresse
        ttk.Label(self.settings_tab, text="IP-Adresse:").pack(pady=5)
        self.ip_entry = ttk.Entry(self.settings_tab)
        self.ip_entry.insert(0, self.config["ip"])  # Initialwert aus config.json
        self.ip_entry.pack(pady=5)

        # Port
        ttk.Label(self.settings_tab, text="Port:").pack(pady=5)
        self.port_entry = ttk.Entry(self.settings_tab)
        self.port_entry.insert(0, self.config["port"])  # Initialwert aus config.json
        self.port_entry.pack(pady=5)

        # Erlaubte Domains
        ttk.Label(self.settings_tab, text="Erlaubte Domains (allow_origins):").pack(pady=5)
        self.origins_entry = ttk.Entry(self.settings_tab)
        self.origins_entry.insert(0, ", ".join(self.config["allow_origins"]))  # Initialwert aus config.json
        self.origins_entry.pack(pady=5)

        # Erlaubte Methoden
        ttk.Label(self.settings_tab, text="Erlaubte Methoden (allow_methods):").pack(pady=5)
        self.methods_entry = ttk.Entry(self.settings_tab)
        self.methods_entry.insert(0, ", ".join(self.config["allow_methods"]))  # Initialwert aus config.json
        self.methods_entry.pack(pady=5)

        # Erlaubte Header
        ttk.Label(self.settings_tab, text="Erlaubte Header (allow_headers):").pack(pady=5)
        self.headers_entry = ttk.Entry(self.settings_tab)
        self.headers_entry.insert(0, ", ".join(self.config["allow_headers"]))  # Initialwert aus config.json
        self.headers_entry.pack(pady=5)

        # Übernehmen-Schaltfläche
        self.save_button = ttk.Button(self.settings_tab, text="Einstellungen speichern", command=self.apply_settings)
        self.save_button.pack(pady=20)


    def update_training_logs(self):
        log_path = "./training_logs/training_logs.txt"
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                logs = f.read()
            self.training_logs_text.config(state="normal")
            self.training_logs_text.delete("1.0", tk.END)
            self.training_logs_text.insert(tk.END, logs)
            self.training_logs_text.config(state="disabled")
        else:
            self.training_logs_text.config(state="normal")
            self.training_logs_text.delete("1.0", tk.END)
            self.training_logs_text.insert(tk.END, "Keine Logs verfügbar.")
            self.training_logs_text.config(state="disabled")



    def start_training(self):
        def training_task():
            try:
                # Beispiel: Parameter von Eingabefeldern lesen
                epochs = int(self.epochs_entry.get())
                lr = float(self.lr_entry.get())
                batch_size = int(self.batch_entry.get())
                
                # Dummy-Training (ersetze mit echter Training-Logik)
                from tqdm import tqdm
                for i in tqdm(range(epochs), desc="Training läuft"):
                    self.progress_bar["value"] = (i + 1) * (100 / epochs)
                    self.root.update_idletasks()
                
                self.update_training_logs()  # Logs anzeigen
                messagebox.showinfo("Erfolg", "Training abgeschlossen!")
                self.config["epochs"] = int(self.epochs_entry.get())
                self.config["learning_rate"] = float(self.lr_entry.get())
                self.config["batch_size"] = int(self.batch_entry.get())
                self.save_config()

            except Exception as e:
                messagebox.showerror("Fehler", f"Training fehlgeschlagen: {e}")

        threading.Thread(target=training_task, daemon=True).start()



# Anwendung starten
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
