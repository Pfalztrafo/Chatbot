import tkinter as tk
from tkinter import ttk, messagebox
import os
import threading    # für die API, Trainer
import uvicorn      # für die API
import time         # für die API
import json         # für die Konfiguration
import psutil
import requests
import torch

from train_model import main as start_training_process  # Importiere die main-Funktion aus train_model.py
from api_main import app                                # Importiere die FastAPI-App
from chatbot_main import get_response, save_chat_to_txt # zum chatten


# Systemressourcen-Manager
class SystemResourceManager:
    @staticmethod
    def get_cpu_usage():
        """Gibt die aktuelle CPU-Auslastung in Prozent zurück."""
        return psutil.cpu_percent(interval=0.1)

    @staticmethod
    def get_ram_usage():
        """Gibt die aktuelle RAM-Nutzung und den Gesamtspeicher zurück."""
        ram = psutil.virtual_memory()
        return ram.used / (1024**3), ram.total / (1024**3)
    
    @staticmethod
    def get_gpu_info():
        """Gibt die GPU-Speichernutzung und Gesamtspeicher zurück."""
        try:
            if not torch.cuda.is_available():
                return "GPU: Nicht verfügbar"
            
            # Name der GPU
            gpu_properties = torch.cuda.get_device_properties(0)
            total_memory = gpu_properties.total_memory / (1024**3)  # Gesamtspeicher in GB
            used_memory = torch.cuda.memory_allocated(0) / (1024**3)  # Genutzter Speicher in GB

            return f"GPU: {used_memory:.1f} GB von {total_memory:.1f} GB"
        except Exception as e:
            return f"Fehler beim Abrufen der GPU-Daten: {e}"



# Manager-Klassen
class ModelManager:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.current_model = self.config_manager.get_param("MODEL", "MODEL_NAME", "google/flan-t5-base")
        self.available_models = self.config_manager.get_param("MODEL", "available_models", [])

    def get_current_model(self):
        """Gibt das aktuell geladene Modell zurück."""
        return self.current_model

    def get_available_models(self):
        """Gibt die Liste der verfügbaren Modelle zurück."""
        return self.available_models

    def switch_model(self, new_model, load_callback=None):
        """
        Wechselt das aktuelle Modell.
        :param new_model: Der Name des neuen Modells.
        :param load_callback: Optionale Funktion, die während des Ladens aufgerufen wird.
        """
        if new_model == self.current_model:
            return "Das ausgewählte Modell ist bereits geladen."

        # Neues Modell laden
        try:
            self.current_model = new_model
            self.config_manager.set_param("MODEL", "MODEL_NAME", new_model)
            if load_callback:
                load_callback(new_model)  # Ladelogik von außen
            return f"Modell '{new_model}' wurde erfolgreich geladen."
        except Exception as e:
            return f"Fehler beim Laden des Modells: {e}"



# Konfigurationsmanager
class ConfigManager:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Lädt die Konfiguration aus der Datei."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warnung: {self.config_path} nicht gefunden. Standardwerte werden verwendet.")
            return self.default_config()
        except json.JSONDecodeError as e:
            print(f"Fehler beim Parsen der Konfigurationsdatei: {e}. Standardwerte werden verwendet.")
            return self.default_config()

    def default_config(self):
        """Definiert Standardwerte, falls die Konfigurationsdatei fehlt."""
        return {
            "API": {
                "ip": "0.0.0.0",
                "port": 8000,
                "allow_origins": ["*"],
                "allow_methods": ["*"],
                "allow_headers": ["*"]
            },
            "TRAINING": {
                "epochs": 1,
                "learning_rate": 0.0001,
                "batch_size": 1
            },
            "CHAT": {
                "temperature": 0.7,
                "use_fuzzy_matching": False
            }
        }

    def save_config(self):
        """Speichert die aktuelle Konfiguration in die Datei."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Fehler beim Speichern der Konfigurationsdatei: {e}")

    def get_param(self, section, key, default=None):
        """Holt einen Parameter aus einer bestimmten Sektion."""
        return self.config.get(section, {}).get(key, default)

    def set_param(self, section, key, value):
        """Setzt einen Parameter und speichert die Konfiguration."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()

    def get_api_config(self):
        """Gibt die API-Konfiguration zurück."""
        return self.config.get("API", {})

    def get_training_config(self):
        """Gibt die Trainingskonfiguration zurück."""
        return self.config.get("TRAINING", {})

    def get_chat_config(self):
        """Gibt die Chat-Konfiguration zurück."""
        return self.config.get("CHAT", {})
#---------------------------------------------

def start_api_server(self):
    """Startet den FastAPI-Server in einem separaten Thread."""
    def run_server():
        try:
            ssl_config = self.load_ssl_config()
            if ssl_config:
                print(f"SSL-Zertifikate gefunden: {ssl_config['keyfile']} und {ssl_config['certfile']}")
                uvicorn.run(
                    app,
                    host=self.config["ip"],
                    port=self.config["port"],
                    ssl_keyfile=ssl_config["keyfile"],
                    ssl_certfile=ssl_config["certfile"]
                )
            else:
                print("Keine SSL-Zertifikate gefunden. Server läuft im HTTP-Modus.")
                uvicorn.run(app, host=self.config["ip"], port=self.config["port"])
        except Exception as e:
            print(f"Fehler beim Starten der API: {e}")
            self.api_status = "Offline"

    # Server in einem separaten Thread starten
    api_thread = threading.Thread(target=run_server, daemon=True)
    api_thread.start()
    time.sleep(2)  # Kurze Pause, um sicherzustellen, dass der Server startet
    print("API-Server-Thread wurde gestartet.")


# API-Verwaltung
class ChatbotAPI:
    def __init__(self, config_manager):
        """
        Initialisiert die ChatbotAPI-Klasse.
        :param config_manager: Instanz von ConfigManager zum Verwalten der Konfiguration.
        """
        self.config_manager = config_manager
        self.app = None  # Hier wird die FastAPI-App aus api_main importiert
        self.server_thread = None
        self.is_running = False

    def load_app(self):
        """Lädt die FastAPI-App aus api_main."""
        try:
            from api_main import app, load_ssl_config
            self.app = app

            # Überprüfen, ob SSL-Zertifikate gefunden werden
            ssl_config = load_ssl_config()
            if ssl_config:
                print(f"SSL-Zertifikate gefunden: {ssl_config['keyfile']} und {ssl_config['certfile']}")
            else:
                print("Keine SSL-Zertifikate gefunden. Server läuft im HTTP-Modus.")
        except ImportError as e:
            print(f"Fehler beim Laden der FastAPI-App: {e}")
            self.app = None


    def start(self):
        """Startet den FastAPI-Server in einem separaten Thread."""
        if self.is_running:
            print("API läuft bereits.")
            return

        if not self.app:
            self.load_app()

        if not self.app:
            print("FastAPI-App konnte nicht geladen werden. Serverstart abgebrochen.")
            return

        def run_server():
            ip = self.config_manager.get_param("API", "ip", "0.0.0.0")
            port = self.config_manager.get_param("API", "port", 8000)
            try:
                uvicorn.run(self.app, host=ip, port=port, log_level="info")
            except Exception as e:
                print(f"Fehler beim Starten des Servers: {e}")

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        print("API wurde gestartet.")

    def stop(self):
        """Beendet den FastAPI-Server."""
        if not self.is_running or not self.server_thread:
            print("API ist bereits gestoppt.")
            return

        print("API wird gestoppt...")
        self.is_running = False
        # FastAPI bietet keine eingebaute Methode, um den Server zu stoppen.
        # Sie können stattdessen zusätzliche Logik verwenden, um Prozesse zu beenden.
        self.server_thread = None
        print("API wurde gestoppt.")

    def get_status(self):
        """Gibt den aktuellen Status der API zurück."""
        return "Läuft" if self.is_running else "Gestoppt"
#---------------------------------------------

# Trainer-Klasse
class Trainer:
    def __init__(self, config_manager):
        """
        Initialisiert die Trainer-Klasse.
        :param config_manager: Instanz von ConfigManager, um Trainingsparameter zu verwalten.
        """
        self.config_manager = config_manager
        self.training_thread = None
        self.is_training = False
        self.logs = []

    def get_training_config(self):
        """Lädt die aktuellen Trainingsparameter."""
        return self.config_manager.get_training_config()

    def start_training(self, update_logs_callback=None):
        """
        Startet den Trainingsprozess in einem separaten Thread.
        :param update_logs_callback: Optionaler Callback, um Logs während des Trainings zu aktualisieren.
        """
        if self.is_training:
            print("Training läuft bereits.")
            return

        def training_task():
            try:
                self.is_training = True
                training_config = self.get_training_config()
                print(f"Training gestartet mit Parametern: {training_config}")

                # Beispiel: Dummy-Trainingsprozess
                for epoch in range(training_config.get("epochs", 1)):
                    if not self.is_training:
                        break
                    time.sleep(1)  # Simuliert die Trainingszeit
                    log_message = f"Epoch {epoch + 1}/{training_config['epochs']} abgeschlossen."
                    self.logs.append(log_message)
                    print(log_message)
                    if update_logs_callback:
                        update_logs_callback(log_message)

                self.logs.append("Training abgeschlossen.")
                print("Training abgeschlossen.")
                if update_logs_callback:
                    update_logs_callback("Training abgeschlossen.")

            except Exception as e:
                error_message = f"Fehler während des Trainings: {e}"
                self.logs.append(error_message)
                print(error_message)
                if update_logs_callback:
                    update_logs_callback(error_message)

            finally:
                self.is_training = False

        self.training_thread = threading.Thread(target=training_task, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        """Stoppt das Training."""
        if not self.is_training:
            print("Kein Training läuft.")
            return
        print("Training wird gestoppt...")
        self.is_training = False
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join()
        print("Training wurde gestoppt.")

    def get_logs(self):
        """Gibt die Trainingslogs zurück."""
        return self.logs
#---------------------------------------------

# GUI-Klasse
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Management Dashboard")
        self.root.geometry("1000x700")

        # Instanzen der Manager-Klassen
        self.config_manager = ConfigManager()
        self.api_manager = ChatbotAPI(self.config_manager)
        self.trainer = Trainer(self.config_manager)
        self.model_manager = ModelManager(self.config_manager)

        # Variablen für den Status
        self.api_status = tk.StringVar(value="Gestoppt")

        # GUI-Elemente erstellen
        self.create_widgets()

        # Log-Verzeichnis
        self.chat_logs_dir = "chat_logs"
        os.makedirs(self.chat_logs_dir, exist_ok=True)

        # Starte die Systemressourcen- und Chatlog-Updates
        self.update_system_stats()
        self.update_chat_logs()

        # API-Status initial aktualisieren
        self.update_api_status_label()

    # --------------------------------------------------------
    # GUI-Tabs und Widgets
    
    def create_widgets(self):
        """Erstellt die GUI-Tabs und Widgets."""
        # Breite der Tabs nur horizontal anpassen
        style = ttk.Style()
        style.configure('TNotebook.Tab', padding=[60, 5])  # Breite: 60, Höhe: 5 bleibt gleich

        # Notebook erstellen
        tab_control = ttk.Notebook(self.root, style='TNotebook')

        # Tabs hinzufügen
        self.overview_tab = ttk.Frame(tab_control)
        self.chat_tab = ttk.Frame(tab_control)
        self.training_tab = ttk.Frame(tab_control)

        tab_control.add(self.overview_tab, text="Übersicht")
        tab_control.add(self.chat_tab, text="Chatbot-Einstellungen")
        tab_control.add(self.training_tab, text="Training")

        tab_control.pack(expand=1, fill="both")  # Tabs horizontal strecken

        # Inhalte der Tabs erstellen
        self.create_overview_tab()
        self.create_chat_tab()
        self.create_training_tab()



    def create_overview_tab(self):
        """Erstellt den Überblick-Tab."""

        # API-Status
        ttk.Label(self.overview_tab, text="API-Status", font=("Arial", 16)).pack(pady=10)
        self.api_status_label = ttk.Label(self.overview_tab, text="API-Status: Gestoppt", font=("Arial", 12))
        self.api_status_label.pack(pady=5)
        ttk.Button(self.overview_tab, text="API Starten/Stoppen", command=self.toggle_api).pack(pady=10)

        # Chat Logs
        ttk.Label(self.overview_tab, text="Chat Logs", font=("Arial", 16)).pack(pady=10)
        self.chat_logs_text = tk.Text(self.overview_tab, height=15, width=80, state="disabled")
        self.chat_logs_text.pack(pady=10)

        # Starte die Aktualisierung der Chatlogs
        self.update_chat_logs()

        # Horizontale Darstellung der Systemressourcen
        self.cpu_label_overview, self.ram_label_overview, self.gpu_label_overview = self.create_system_resource_labels(self.overview_tab)


    def create_chat_tab(self):
        """Erstellt den Chatbot-Einstellungen-Tab."""
        ttk.Label(self.chat_tab, text="Live-Chat mit dem Bot", font=("Arial", 16)).pack(pady=10)

        # Modellauswahl
        ttk.Label(self.chat_tab, text="Modell auswählen:", font=("Arial", 12)).pack(pady=5)
        self.model_var = tk.StringVar(value=self.model_manager.get_current_model())
        self.model_dropdown = ttk.Combobox(self.chat_tab, textvariable=self.model_var, state="readonly")
        self.model_dropdown['values'] = self.model_manager.get_available_models()
        self.model_dropdown.pack(pady=5)
        ttk.Button(self.chat_tab, text="Modell wechseln", command=self.handle_model_switch).pack(pady=10)

        # Chat-Anzeige
        self.chat_history = tk.Text(self.chat_tab, height=20, width=80, state="disabled")
        self.chat_history.pack(pady=10)

        # Eingabefeld für den Benutzer
        self.chat_input = ttk.Entry(self.chat_tab, width=70)
        self.chat_input.pack(pady=5)
        self.chat_input.bind("<Return>", self.handle_chat_input)

        # Parameter anpassen
        ttk.Label(self.chat_tab, text="Chat-Einstellungen", font=("Arial", 14)).pack(pady=10)
        self.temperature_entry = ttk.Entry(self.chat_tab)
        self.temperature_entry.insert(0, self.config_manager.get_param("CHAT", "temperature", 0.7))
        self.temperature_entry.pack(pady=5)

        ttk.Button(self.chat_tab, text="Einstellungen Übernehmen", command=self.apply_chat_settings).pack(pady=10)
        
        # Horizontale Darstellung der Systemressourcen
        self.cpu_label_chat, self.ram_label_chat, self.gpu_label_chat = self.create_system_resource_labels(self.chat_tab)
    


    def create_training_tab(self):
        """Erstellt den Trainings-Tab."""
        ttk.Label(self.training_tab, text="Training Parameter", font=("Arial", 16)).pack(pady=10)

        # Modellauswahl
        ttk.Label(self.training_tab, text="Modell für Training auswählen:", font=("Arial", 12)).pack(pady=5)
        self.training_model_var = tk.StringVar(value=self.model_manager.get_current_model())
        self.training_model_dropdown = ttk.Combobox(self.training_tab, textvariable=self.training_model_var, state="readonly")
        self.training_model_dropdown['values'] = self.model_manager.get_available_models()
        self.training_model_dropdown.pack(pady=5)

        # Parameterfelder
        self.epochs_entry = ttk.Entry(self.training_tab)
        self.epochs_entry.insert(0, self.config_manager.get_param("TRAINING", "epochs", 1))
        self.epochs_entry.pack(pady=5)

        self.lr_entry = ttk.Entry(self.training_tab)
        self.lr_entry.insert(0, self.config_manager.get_param("TRAINING", "learning_rate", 0.0001))
        self.lr_entry.pack(pady=5)

        self.batch_size_entry = ttk.Entry(self.training_tab)
        self.batch_size_entry.insert(0, self.config_manager.get_param("TRAINING", "batch_size", 32))
        self.batch_size_entry.pack(pady=5)

        # Logs und Fortschrittsanzeige
        ttk.Label(self.training_tab, text="Trainingslogs", font=("Arial", 14)).pack(pady=10)
        self.training_logs_text = tk.Text(self.training_tab, height=15, width=80, state="disabled")
        self.training_logs_text.pack(pady=10)

        # Trainingsbuttons
        ttk.Button(self.training_tab, text="Training Starten", command=self.start_training).pack(pady=5)
        ttk.Button(self.training_tab, text="Training Stoppen", command=self.stop_training).pack(pady=5)

        # Horizontale Darstellung der Systemressourcen
        self.cpu_label_training, self.ram_label_training, self.gpu_label_training = self.create_system_resource_labels(self.training_tab)

    # --------------------------------------------------------
    # Funktionen
    # Systemressourcen-Labels
    def create_system_resource_labels(self, parent):
        """Erstellt die Labels für CPU, RAM und GPU horizontal in einem Parent-Widget."""
        # Erstelle einen Container-Frame
        resource_frame = ttk.Frame(parent)
        resource_frame.pack(side="bottom", pady=10, fill="x")  # Unten platzieren, horizontal strecken

        # Systemressourcen-Titel
        title_label = ttk.Label(resource_frame, text="Systemressourcen:", font=("Arial", 12, "bold"), anchor="w")
        title_label.pack(side="left", padx=10)  # Linksbündig

        # CPU-Label
        cpu_label = ttk.Label(resource_frame, text="CPU: 0%", font=("Arial", 12))
        cpu_label.pack(side="left", padx=10)

        # RAM-Label
        ram_label = ttk.Label(resource_frame, text="RAM: 0 GB von 0 GB", font=("Arial", 12))
        ram_label.pack(side="left", padx=10)

        # GPU-Label
        gpu_label = ttk.Label(resource_frame, text="GPU: Nicht verfügbar", font=("Arial", 12))
        gpu_label.pack(side="left", padx=10)

        # Version
        title_label = ttk.Label(resource_frame, text="Version 1.0.0", font=("Arial", 12, "italic"), anchor="w")
        title_label.pack(side="right", padx=10)  # Rechtsbündig

        return cpu_label, ram_label, gpu_label



    # Modelwechsel
    def handle_model_switch(self):
        """Handhabt den Wechsel des Modells."""
        new_model = self.model_var.get()
        message = self.model_manager.switch_model(new_model)
        messagebox.showinfo("Modellwechsel", message)

    # --------------------------------------------------------
    # Callbacks und Interaktionen
    def toggle_api(self):
        """Startet oder stoppt die API."""
        if self.api_manager.is_running:
            self.api_manager.stop()
        else:
            self.api_manager.start()
        self.update_api_status_label()

    def update_api_status_label(self):
        """Aktualisiert den API-Status in der Übersicht."""
        status = self.api_manager.get_status()
        self.api_status_label.config(text=f"API-Status: {status}")

    def apply_chat_settings(self):
        """Speichert und übernimmt die Chat-Einstellungen."""
        temperature = float(self.temperature_entry.get())
        self.config_manager.set_param("CHAT", "temperature", temperature)
        messagebox.showinfo("Einstellungen", "Chat-Einstellungen gespeichert.")

    def handle_chat_input(self, event):
        """Verarbeitet die Eingabe des Benutzers und gibt eine Antwort des Chatbots zurück."""
        user_input = self.chat_input.get().strip()
        if not user_input:
            return
        self.chat_input.delete(0, tk.END)
        self.append_to_chat("Du", user_input)

        # Antwort abrufen
        from chatbot_main import get_response
        response = get_response(user_input)
        self.append_to_chat("Bot", response)

    def append_to_chat(self, sender, message):
        """Fügt eine Nachricht zum Chat-Verlauf hinzu."""
        self.chat_history.config(state="normal")
        self.chat_history.insert("end", f"{sender}: {message}\n")
        self.chat_history.config(state="disabled")
        self.chat_history.see("end")

    def start_training(self):
        """Startet das Training."""
        self.trainer.start_training(update_logs_callback=self.update_training_logs)

    def stop_training(self):
        """Stoppt das Training."""
        self.trainer.stop_training()

    def update_training_logs(self, log_message):
        """Aktualisiert die Trainingslogs in der GUI."""
        self.training_logs_text.config(state="normal")
        self.training_logs_text.insert("end", f"{log_message}\n")
        self.training_logs_text.config(state="disabled")
        self.training_logs_text.see("end")

    def update_system_stats(self):
        """Aktualisiert die Systemressourcenanzeige."""
        cpu_usage = SystemResourceManager.get_cpu_usage()
        used_ram, total_ram = SystemResourceManager.get_ram_usage()
        gpu_info = SystemResourceManager.get_gpu_info()

        # CPU, RAM und GPU in allen Tabs aktualisieren
        self.cpu_label_overview.config(text=f"CPU: {cpu_usage:.1f}%")
        self.ram_label_overview.config(text=f"RAM: {used_ram:.1f} GB von {total_ram:.1f} GB")
        self.gpu_label_overview.config(text=gpu_info)

        self.cpu_label_chat.config(text=f"CPU: {cpu_usage:.1f}%")
        self.ram_label_chat.config(text=f"RAM: {used_ram:.1f} GB von {total_ram:.1f} GB")
        self.gpu_label_chat.config(text=gpu_info)

        self.cpu_label_training.config(text=f"CPU: {cpu_usage:.1f}%")
        self.ram_label_training.config(text=f"RAM: {used_ram:.1f} GB von {total_ram:.1f} GB")
        self.gpu_label_training.config(text=gpu_info)

        # Wiederhole die Aktualisierung alle 1 Sekunde
        self.root.after(1000, self.update_system_stats)

    # Chatlogs aktualisieren
    def update_chat_logs(self):
        """Prüft regelmäßig auf neue Chatlogs und aktualisiert die Anzeige im Übersicht-Tab."""
        try:
            # Lade die neuesten Logs aus dem Verzeichnis
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
                self.chat_logs_text.config(state="normal")
                self.chat_logs_text.delete("1.0", tk.END)
                self.chat_logs_text.insert(tk.END, logs)
                self.chat_logs_text.see(tk.END)  # Automatisch ans Ende scrollen
                self.chat_logs_text.config(state="disabled")
            else:
                self.chat_logs_text.config(state="normal")
                self.chat_logs_text.delete("1.0", tk.END)
                self.chat_logs_text.insert(tk.END, "Keine Logs verfügbar.")
                self.chat_logs_text.see(tk.END)
                self.chat_logs_text.config(state="disabled")
        except Exception as e:
            print(f"Fehler beim Aktualisieren der Logs: {e}")

        # Logs alle 5 Sekunden aktualisieren
        self.root.after(5000, self.update_chat_logs)



# --------------------------------------------------------
# Anwendung starten
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
