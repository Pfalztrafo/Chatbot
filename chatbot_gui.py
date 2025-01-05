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

import chatbot_main
from train_model import main as start_training_process  # Importiere die main-Funktion aus train_model.py
from api_main import app, load_ssl_config               # Importiere die FastAPI-App
from chatbot_main import init_chatbot, get_response  # Ganz oben

import subprocess
import sys
import shutil

# Systemressourcen-Manager
def get_gpu_info_global():
    try:
        if not torch.cuda.is_available():
            return "GPU: Nicht verfügbar"
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
        used, total = result.strip().split(',')
        used = float(used)
        total = float(total)
        return f"GPU: {used:.1f} MB von {total:.1f} MB"
    except Exception as e:
        return f"Fehler beim Abrufen GPU: {e}"


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
        """
        Liest die globale GPU-Speichernutzung für GPU 0 via nvidia-smi aus
        und gibt sie in GB an.
        """
        # 1) Prüfen, ob nvidia-smi überhaupt vorhanden ist
        if shutil.which("nvidia-smi") is None:
            return "GPU: nvidia-smi nicht verfügbar"

        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i", "0"   # GPU 0 abfragen, ggf. anpassen bei mehreren GPUs
                ],
                text=True
            )
            used_mb, total_mb = result.strip().split(',')
            used_mb = float(used_mb)
            total_mb = float(total_mb)

            # 3) Umrechnung MB -> GB
            used_gb = used_mb / 1024.0
            total_gb = total_mb / 1024.0

            return f"GPU: {used_gb:.1f} GB von {total_gb:.1f} GB"
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
            from api_main import load_ssl_config
            ssl_conf = load_ssl_config()
            if ssl_conf:
                uvicorn.run(self.app, host=ip, port=port,
                            ssl_keyfile=ssl_conf["keyfile"],
                            ssl_certfile=ssl_conf["certfile"],
                            log_level="info")
            else:
                uvicorn.run(self.app, host=ip, port=port, log_level="info")

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
        self.server_thread = None
        print("API wurde gestoppt.")

    def get_status(self):
        """Gibt den aktuellen Status der API zurück."""
        return "Läuft" if self.is_running else "Gestoppt"
#---------------------------------------------

# Trainer-Klasse
import subprocess
import threading

class Trainer:
    def __init__(self, config_manager):
        """
        Trainer-Klasse zum Aufrufen von train_model.py in einem separaten Thread.
        Mit is_training = False/True kannst du in der GUI abfragen,
        ob das Training läuft.
        """
        self.config_manager = config_manager
        self.is_training = False
        self.logs = []

    def start_training(self, update_logs_callback=None):
        """
        Startet den Trainingsprozess in einem separaten Thread,
        indem train_model.py als Subprozess ausgeführt wird.

        :param update_logs_callback: (optional) Eine Funktion, die aufgerufen wird,
                                     wenn das Training abgeschlossen ist oder ein Fehler auftritt.
                                     Kann z.B. ein GUI-Callback sein, um Statuslabel zu aktualisieren.
        """
        if self.is_training:
            print("Training läuft bereits.")
            return

        def training_task():
            try:
                self.is_training = True

                # Starte Subprozess unbuffered (-u), ohne stdout-Pipe => Output landet direkt im Terminal.
                process = subprocess.Popen(
                    ["python", "-u", "train_model.py"],
                    stdout=None,
                    stderr=None,
                    text=True
                )

                # Warte, bis der Subprozess beendet ist
                process.wait()

                if process.returncode == 0:
                    # Training erfolgreich abgeschlossen
                    self.logs.append("Training abgeschlossen.")
                    print("Training abgeschlossen.")
                    if update_logs_callback:
                        update_logs_callback()  # z.B. callback("done")
                else:
                    # Fehlerhafter Exit-Code
                    error_msg = f"Fehler: train_model.py gab Returncode {process.returncode}"
                    self.logs.append(error_msg)
                    print(error_msg)
                    if update_logs_callback:
                        update_logs_callback()  # z.B. callback("error")
            except Exception as e:
                # Unerwarteter Fehler beim Starten oder Ausführen
                error_message = f"Fehler während des Trainings: {e}"
                self.logs.append(error_message)
                print(error_message)
                if update_logs_callback:
                    update_logs_callback()  # z.B. callback("error")
            finally:
                self.is_training = False

        # Thread starten (daemon=True, damit er beim Schließen des Hauptprogramms endet)
        training_thread = threading.Thread(target=training_task, daemon=True)
        training_thread.start()

    def get_logs(self):
        """
        Gibt die bisher gesammelten Logs zurück (z.B. Fehlermeldungen).
        """
        return self.logs

#---------------------------------------------

# GUI-Klasse
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Management Dashboard")
        self.root.geometry("1000x900")

        # Der Chatbot soll zu Beginn "Gestoppt" sein
        self.chatbot_status_var = tk.StringVar(value="Gestoppt")


        # Instanzen der Manager-Klassen
        self.config_manager = ConfigManager()
        self.api_manager = ChatbotAPI(self.config_manager)
        self.trainer = Trainer(self.config_manager)
        self.model_manager = ModelManager(self.config_manager)

        # Variablen für den Status
        self.api_status = tk.StringVar(value="Gestoppt")

        # Log-Verzeichnis
        self.chat_logs_dir = "chat_logs"
        os.makedirs(self.chat_logs_dir, exist_ok=True)

        # GUI-Elemente erstellen
        self.create_widgets()

        # Starte die Systemressourcen- und Chatlog-Updates
        self.update_system_stats()
        self.update_chat_logs()

        # API starten
        self.start_api_server()

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
        """Erstellt den Überblick-Tab mit API-Status (links) und Chatlog (rechts) im grid-Layout
        und sorgt für ein einheitliches Design wie in den anderen Tabs."""

        # Hauptcontainer (ähnlich wie main_frame in anderen Tabs)
        main_frame = ttk.Frame(self.overview_tab)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Die Übersicht-Tab soll sich ausdehnen
        self.overview_tab.grid_rowconfigure(0, weight=1)
        self.overview_tab.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)  # linke Spalte (API-Status)
        main_frame.grid_columnconfigure(1, weight=3)  # rechte Spalte (Chatlog)

        # -------------------------
        # Linke Spalte (API-Status, Einstellungen, Chatbot-Schalter)
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Überschrift
        ttk.Label(left_frame, text="API- und Chatbot-Steuerung", font=("Arial", 12))\
            .grid(row=0, column=0, columnspan=2, pady=10, sticky="w")

        # 1) API-Status
        self.api_status_label = ttk.Label(left_frame, text="Gestoppt", font=("Arial", 10))
        self.api_status_label.grid(row=1, column=0, pady=(0, 5), sticky="w")

        # 2) Button: API Starten / Stoppen
        ttk.Button(
            left_frame,
            text="API Starten/Stoppen",
            command=self.toggle_api
        ).grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        # 3) IP, Port & Domains
        ttk.Label(left_frame, text="IP-Adresse:", font=("Arial", 10))\
            .grid(row=3, column=0, sticky="w", pady=(0, 5))
        self.ip_entry = ttk.Entry(left_frame, font=("Arial", 10), width=18)
        self.ip_entry.insert(0, self.config_manager.get_param("API", "ip", "0.0.0.0"))
        self.ip_entry.grid(row=3, column=1, pady=(0, 5), sticky="w")

        ttk.Label(left_frame, text="Port:", font=("Arial", 10))\
            .grid(row=4, column=0, sticky="w", pady=(0, 5))
        self.port_entry = ttk.Entry(left_frame, font=("Arial", 10), width=18)
        self.port_entry.insert(0, self.config_manager.get_param("API", "port", 8000))
        self.port_entry.grid(row=4, column=1, pady=(0, 5), sticky="w")

        ttk.Label(left_frame, text="Domains:", font=("Arial", 10))\
            .grid(row=5, column=0, sticky="w", pady=(0, 5))
        self.origins_entry = ttk.Entry(left_frame, font=("Arial", 10), width=18)
        self.origins_entry.insert(0, ", ".join(self.config_manager.get_param("API", "allow_origins", ["*"])))
        self.origins_entry.grid(row=5, column=1, pady=(0, 5), sticky="w")

        ttk.Button(
            left_frame,
            text="Einstellungen speichern",
            command=self.save_api_changes
        ).grid(row=6, column=0, columnspan=2, pady=10, sticky="ew")

        # 4) Chatbot An/Aus
        ttk.Button(
            left_frame,
            text="Chatbot An/Aus",
            command=self.toggle_chatbot
        ).grid(row=7, column=0, columnspan=2, pady=10, sticky="ew")

        # Status-Label für Chatbot
        self.chatbot_status_label_overview = ttk.Label(left_frame, text="Chatbot: Aus", foreground="red", font=("Arial", 10))
        self.chatbot_status_label_overview.grid(row=8, column=0, columnspan=2, pady=(0, 10), sticky="w")

        # 5) Beenden-Button
        ttk.Button(
            left_frame,
            text="Beenden",
            command=self.quit_application  # z.B. self.root.quit oder destroy
        ).grid(row=9, column=0, columnspan=2, pady=10, sticky="ew")

        # -------------------------
        # Rechte Spalte (Chat Logs)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ttk.Label(right_frame, text="Chat Logs", font=("Arial", 12))\
            .grid(row=0, column=0, pady=10, sticky="w")

        self.chat_logs_text = tk.Text(right_frame, wrap="word", state="disabled", font=("Arial", 10))
        self.chat_logs_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # 6) Ressourcen-Anzeige unten (wie in den anderen Tabs)
        self.cpu_label_overview, self.ram_label_overview, self.gpu_label_overview = self.create_system_resource_labels(self.overview_tab)


    def save_api_changes(self):
        """Speichert die Änderungen an den API-Einstellungen."""
        try:
            new_ip = self.ip_entry.get()
            new_port = int(self.port_entry.get())
            new_origins = [origin.strip() for origin in self.origins_entry.get().split(",")]

            self.config_manager.set_param("API", "ip", new_ip)
            self.config_manager.set_param("API", "port", new_port)
            self.config_manager.set_param("API", "allow_origins", new_origins)

            messagebox.showinfo("Erfolg", "Einstellungen wurden gespeichert.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Einstellungen konnten nicht gespeichert werden: {e}")

    # --------------------------------------------------------

    def create_chat_tab(self):
        """Erstellt den Chat-Tab mit Modellauswahl, Parametereinstellungen links und Chat-Verlauf rechts."""
        
        # Hauptcontainer, der alles in Grid verwaltet
        main_frame = ttk.Frame(self.chat_tab)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.chat_tab.grid_rowconfigure(0, weight=1)
        self.chat_tab.grid_columnconfigure(0, weight=1)

        # Zwei Spalten im main_frame:
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=4)  
        main_frame.grid_columnconfigure(1, weight=1)

        # ------------------------------------------------------------
        # 1) Linke Spalte (Parameter) mit Scrollbar
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        canvas = tk.Canvas(left_frame)
        scrollbar_left = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # Frame im Canvas scrollbar
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_left.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_left.pack(side="right", fill="y")

        # Grid für unser scrollable_frame
        scrollable_frame.grid_rowconfigure(0, weight=0)
        scrollable_frame.grid_columnconfigure(0, weight=0)
        scrollable_frame.grid_columnconfigure(1, weight=1)

        # -- Parameter-Widgets im scrollable_frame
        row = 0
        ttk.Label(scrollable_frame, text="Modell für Chat auswählen:", font=("Arial", 12))\
            .grid(row=row, column=0, columnspan=2, pady=2, sticky="w")
        row += 1

        self.chat_model_var = tk.StringVar(value=self.model_manager.get_current_model())
        self.chat_model_dropdown = ttk.Combobox(
            scrollable_frame,
            textvariable=self.chat_model_var,
            state="readonly"
        )
        self.chat_model_dropdown['values'] = self.model_manager.get_available_models()
        self.chat_model_dropdown.grid(row=row, column=0, columnspan=2, pady=5, sticky="ew")
        row += 1

        parameters = [
            ("Fuzzy Matching:", "use_fuzzy_matching", "bool"),
            ("Generative KI:", "use_ki_generative", "bool"),
            ("Pipeline nutzen:", "use_pipeline", "bool"),
            ("Interner Prompt:", "internal_prompt", "str"),
            ("Fuzzy Threshold:", "fuzzy_threshold", "int"),
            ("Fuzzy Score Range:", "fuzzy_score_range", "range"),
            ("BLEU Score Range:", "bleu_score_range", "range"),
            ("ROUGE-L Score Range:", "rougeL_score_range", "range"),
            ("Log Score Threshold:", "log_score_threshold", "float"),
            ("Sampling:", "do_sample", "bool"),
            ("Temperature:", "temperature", "float"),
            ("Top-K:", "top_k", "int"),
            ("Top-P:", "top_p", "float"),
            ("Num Beams:", "num_beams", "int"),
            ("Repetition Penalty:", "repetition_penalty", "float"),
            ("Maximale Länge:", "max_length", "int"),
            ("Minimale Länge:", "min_length", "int"),
            ("No Repeat N-Gram Size:", "no_repeat_ngram_size", "int"),
            ("Längenstrafe:", "length_penalty", "float"),
            ("Frühes Stoppen:", "early_stopping", "bool"),
        ]

        self.chat_entries = {}

        for label_text, param, param_type in parameters:
            # Label
            lbl = ttk.Label(scrollable_frame, text=label_text, font=("Arial", 10))
            lbl.grid(row=row, column=0, pady=5, sticky="w")

            if param_type == "bool":
                var = tk.BooleanVar(value=self.config_manager.get_param("CHAT", param, False))
                cb = ttk.Checkbutton(scrollable_frame, variable=var)
                cb.grid(row=row, column=1, pady=5, sticky="w")
                self.chat_entries[param] = var

            elif param_type == "range":
                fr = ttk.Frame(scrollable_frame)
                fr.grid(row=row, column=1, pady=5, sticky="w")
                default_range = self.config_manager.get_param("CHAT", param, [0.0, 1.0])

                min_entry = ttk.Entry(fr, width=10)
                min_entry.insert(0, default_range[0])
                min_entry.pack(side="left")

                ttk.Label(fr, text=" - ").pack(side="left")

                max_entry = ttk.Entry(fr, width=10)
                max_entry.insert(0, default_range[1])
                max_entry.pack(side="left")

                self.chat_entries[param] = (min_entry, max_entry)

            elif param_type == "str" and param == "internal_prompt":
                textw = tk.Text(scrollable_frame, height=10, width=20, wrap="word")  # Höhe erhöht, Breite reduziert
                textw.insert("1.0", self.config_manager.get_param("CHAT", param, ""))
                textw.grid(row=row, column=1, pady=5, sticky="n")  # Nur vertikales Ausrichten
                self.chat_entries[param] = textw

            else:
                entry = ttk.Entry(scrollable_frame)
                entry.insert(0, self.config_manager.get_param("CHAT", param, ""))
                entry.grid(row=row, column=1, pady=5, sticky="ew")
                self.chat_entries[param] = entry

            row += 1

        # Button: Einstellungen speichern
        ttk.Button(scrollable_frame, text="Einstellungen Übernehmen", command=self.save_chat_settings)\
            .grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
        row += 1

        # Chatbot An/Aus-Button
        ttk.Button(
            scrollable_frame, 
            text="Chatbot An/Aus", 
            command=self.toggle_chatbot
        ).grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")

        row += 1

        self.chatbot_status_label_chat = ttk.Label(scrollable_frame, text="Chatbot: Aus", foreground="red", font=("Arial", 12))
        self.chatbot_status_label_chat.grid(row=row, column=0, columnspan=2, pady=5, sticky="w")

        row += 1

        # ------------------------------------------------------------
        # 2) Rechte Spalte (Chat-Verlauf)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Rechte Spalte anpassen
        ttk.Label(right_frame, text="Chat-Verlauf", font=("Arial", 12))\
            .grid(row=0, column=0, sticky="w", pady=(0, 10))

        # Scrollbar für den Chat-Verlauf
        scrollbar_chat = ttk.Scrollbar(right_frame, orient="vertical")
        scrollbar_chat.grid(row=1, column=1, sticky="ns")

        self.chat_history = tk.Text(
            right_frame, wrap="word", state="disabled",
            font=("Arial", 10),
            width=70,  # Reduziert die Breite
            yscrollcommand=scrollbar_chat.set
        )
        self.chat_history.grid(row=1, column=0, sticky="nsew", padx=(0,5), pady=5)

        scrollbar_chat.config(command=self.chat_history.yview)

        # Eingabezeile + Send-Button
        input_frame = ttk.Frame(right_frame)
        input_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)

        self.chat_input = ttk.Entry(input_frame, font=("Arial", 10))
        self.chat_input.pack(side="left", fill="x", expand=True, padx=5)
        self.chat_input.bind("<Return>", self.handle_chat_input)

        send_button = ttk.Button(input_frame, text="➤", command=self.handle_chat_input)
        send_button.pack(side="right")

        # Rechte Spalte dehnbar machen
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Systemressourcen-Anzeige
        self.cpu_label_chat, self.ram_label_chat, self.gpu_label_chat = self.create_system_resource_labels(self.chat_tab)


    def save_chat_settings(self):
        """Speichert die geänderten Chat-Parameter in config.json."""
        try:
            for param, widget in self.chat_entries.items():
                if isinstance(widget, tk.BooleanVar):
                    value = widget.get()

                elif isinstance(widget, tuple):  # Range
                    # Die Range wird als Liste von zwei Werten gespeichert
                    value = [float(widget[0].get()), float(widget[1].get())]

                elif isinstance(widget, tk.Text):  # Textfeld
                    value = widget.get("1.0", "end").strip()

                else:
                    # Normaler Entry-Fall
                    value = widget.get().strip()

                    # Typkonvertierung anhand des Param-Namens
                    if param in ["temperature", "top_p", "repetition_penalty", 
                                "length_penalty", "log_score_threshold"]:
                        value = float(value)
                    elif param in ["fuzzy_threshold", "top_k", "num_beams", 
                                "max_length", "min_length", 
                                "no_repeat_ngram_size"]:
                        value = int(value)

                # Wert in die config schreiben
                self.config_manager.set_param("CHAT", param, value)

                # In save_chat_settings() oder am Ende von create_chat_tab():
                selected_model = self.chat_model_var.get()
                self.config_manager.set_param("MODEL", "MODEL_NAME", selected_model)

            messagebox.showinfo("Erfolg", "Chat-Parameter erfolgreich gespeichert.")
        except ValueError as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern der Chat-Parameter: {e}")

    # --------------------------------------------------------

    def create_training_tab(self):
        """Erstellt den Trainings-Tab mit Modellauswahl links und Logs rechts."""

        # Hauptcontainer für Parameter und Logs
        main_frame = ttk.Frame(self.training_tab)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Konfiguration des Grid-Layouts für den Tab
        self.training_tab.grid_rowconfigure(0, weight=1)
        self.training_tab.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)  # Linke Spalte
        main_frame.grid_columnconfigure(1, weight=3)  # Rechte Spalte

        # Linke Spalte: Modellauswahl und Parameter
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="ns", padx=10, pady=10)

        # Modellauswahl
        ttk.Label(left_frame, text="Modell für Training auswählen:", font=("Arial", 12)).grid(row=0, column=0, columnspan=2, pady=5, sticky="w")
        self.training_model_var = tk.StringVar(value=self.model_manager.get_current_model())
        self.training_model_dropdown = ttk.Combobox(left_frame, textvariable=self.training_model_var, state="readonly")
        self.training_model_dropdown['values'] = self.model_manager.get_available_models()
        self.training_model_dropdown.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        # Trainingsparameter
        parameters = [
            ("Epochen:", "epochs", "int"),
            ("Lernrate:", "learning_rate", "float"),
            ("Batchgröße:", "batch_size", "int"),
            ("Weight Decay:", "weight_decay", "float"),
            ("Trainingsverhältnis:", "train_ratio", "float"),
            ("Negative Sample Rate:", "negative_sample_rate", "float"),
        ]

        self.training_entries = {}

        for idx, (label, param, param_type) in enumerate(parameters, start=2):
            ttk.Label(left_frame, text=label, font=("Arial", 10)).grid(row=idx, column=0, pady=5, sticky="w")
            entry = ttk.Entry(left_frame, font=("Arial", 10))
            default_value = self.config_manager.get_param("TRAINING", param, "")
            entry.insert(0, str(default_value))
            entry.grid(row=idx, column=1, pady=5, sticky="ew")
            self.training_entries[param] = (entry, param_type)

        # Trainingsargumente (training_args)
        ttk.Label(left_frame, text="Trainingsargumente:", font=("Arial", 12)).grid(row=8, column=0, columnspan=2, pady=10, sticky="w")

        training_args_params = [
            ("Eval Strategy:", "eval_strategy", "str", ["epoch", "steps", "no"]),
            ("Save Strategy:", "save_strategy", "str", ["epoch", "steps", "no"]),
            ("Load Best Model at End:", "load_best_model_at_end", "bool", None),
            ("Save Total Limit:", "save_total_limit", "int", None),
            ("Logging Steps:", "logging_steps", "int", None),
            ("Early Stopping Patience:", "early_stopping_patience", "int", None),
        ]

        self.training_args_entries = {}

        for idx, (label, param, param_type, options) in enumerate(training_args_params, start=9):
            ttk.Label(left_frame, text=label, font=("Arial", 10)).grid(row=idx, column=0, pady=5, sticky="w")
            
            if param_type == "bool":
                var = tk.BooleanVar(value=self.config_manager.get_param("TRAINING", "training_args", {}).get(param, False))
                cb = ttk.Checkbutton(left_frame, variable=var)
                cb.grid(row=idx, column=1, pady=5, sticky="w")
                self.training_args_entries[param] = var

            elif param_type == "str" and options:
                var = tk.StringVar(value=self.config_manager.get_param("TRAINING", "training_args", {}).get(param, options[0]))
                dropdown = ttk.Combobox(left_frame, textvariable=var, state="readonly", values=options)
                dropdown.grid(row=idx, column=1, pady=5, sticky="ew")
                self.training_args_entries[param] = var

            else:
                entry = ttk.Entry(left_frame, font=("Arial", 10))
                default_value = self.config_manager.get_param("TRAINING", "training_args", {}).get(param, "")
                entry.insert(0, str(default_value))
                entry.grid(row=idx, column=1, pady=5, sticky="ew")
                self.training_args_entries[param] = (entry, param_type)

        # Checkbox für "Include German Data"
        ttk.Label(left_frame, text="Include German Data:", font=("Arial", 10)).grid(row=15, column=0, pady=5, sticky="w")
        self.include_german_var = tk.BooleanVar(value=self.config_manager.get_param("TRAINING", "include_german", False))
        include_german_checkbox = ttk.Checkbutton(left_frame, variable=self.include_german_var)
        include_german_checkbox.grid(row=15, column=1, pady=5, sticky="w")

        # Buttons für Einstellungen und Training
        ttk.Button(left_frame, text="Einstellungen Übernehmen", command=self.save_training_settings)\
            .grid(row=16, column=0, columnspan=2, pady=10, sticky="ew")
        ttk.Button(left_frame, text="Training Starten", command=self.start_training)\
            .grid(row=17, column=0, columnspan=2, pady=10, sticky="ew")

        # Training-Status-Label (unten im left_frame)
        self.training_status_label = ttk.Label(left_frame, text="Training: Inaktiv", foreground="red", font=("Arial", 12))
        self.training_status_label.grid(row=18, column=0, columnspan=2, pady=5, sticky="w")

        # Chatbot An/Aus-Button
        ttk.Button(
            left_frame, 
            text="Chatbot An/Aus", 
            command=self.toggle_chatbot
        ).grid(row=19, column=0, columnspan=2, pady=10, sticky="ew")

        # Chatbot-Status label
        self.chatbot_status_label_training = ttk.Label(left_frame, text="Chatbot: Aus", foreground="red", font=("Arial", 12))
        self.chatbot_status_label_training.grid(row=20, column=0, columnspan=2, pady=5, sticky="w")

        # Rechte Spalte: Trainingslogs
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ttk.Label(right_frame, text="Trainingslogs", font=("Arial", 12)).grid(row=0, column=0, pady=10, sticky="w")
        self.training_logs_text = tk.Text(right_frame, wrap="word", state="disabled", font=("Arial", 10))
        self.training_logs_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Rechte Spalte dehnbar machen
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Starte die Log-Aktualisierung
        self.update_training_logs()

        # Systemressourcen-Anzeige
        self.cpu_label_training, self.ram_label_training, self.gpu_label_training = self.create_system_resource_labels(self.training_tab)



    def training_finished(self, status=None):
        """
        Wird vom Trainer aufgerufen, sobald das Training abgeschlossen ist 
        oder ein Fehler auftrat.
        """
        if status == "done":
            self.training_status_label.config(text="Training: Abgeschlossen", foreground="blue")
        elif status == "error":
            self.training_status_label.config(text="Training: Fehler", foreground="red")
        else:
            self.training_status_label.config(text="Training: Abgeschlossen", foreground="blue")


    # Trainingseinstellungen
    def save_training_settings(self):
        """Speichert die geänderten Trainingsparameter in config.json."""
        try:
            # Speichern der Standard-TRAINING-Parameter
            for param, (entry, param_type) in self.training_entries.items():
                raw_value = entry.get().strip()
                if param_type == "int":
                    value = int(raw_value)
                elif param_type == "float":
                    value = float(raw_value)
                else:
                    value = raw_value  # Default: string
                self.config_manager.set_param("TRAINING", param, value)

            # Speichern der Trainingsargumente (training_args)
            training_args = {}
            for param, widget in self.training_args_entries.items():
                if isinstance(widget, tk.BooleanVar):
                    value = widget.get()
                elif isinstance(widget, tk.StringVar):
                    value = widget.get()
                else:
                    entry, param_type = widget
                    raw_value = entry.get().strip()
                    if param_type == "int":
                        value = int(raw_value)
                    elif param_type == "float":
                        value = float(raw_value)
                    else:
                        value = raw_value  # Default: string
                training_args[param] = value
            self.config_manager.set_param("TRAINING", "training_args", training_args)

            # Speichern des "Include German Data" Parameters
            self.config_manager.set_param("TRAINING", "include_german", self.include_german_var.get())

            # Speichern der Modellwahl für das Training
            selected_train_model = self.training_model_var.get()
            self.config_manager.set_param("MODEL", "MODEL_NAME", selected_train_model)

            # Erfolgsmeldung
            messagebox.showinfo("Einstellungen gespeichert", "Die Trainingsparameter wurden erfolgreich gespeichert.")
            print("Trainingsparameter erfolgreich gespeichert.")

        except ValueError as e:
            print(f"Fehler beim Speichern der Trainingsparameter: {e}")
            messagebox.showerror("Fehler", f"Fehler beim Speichern der Trainingsparameter: {e}")
        except Exception as e:
            print(f"Unerwarteter Fehler beim Speichern der Trainingsparameter: {e}")
            messagebox.showerror("Fehler", f"Unerwarteter Fehler beim Speichern der Trainingsparameter: {e}")


    # --------------------------------------------------------
    # Funktionen
    # Systemressourcen-Labels
    def create_system_resource_labels(self, parent):
        """Erstellt die Labels für CPU, RAM und GPU horizontal mit `grid`."""
        # Erstelle einen Container-Frame
        resource_frame = ttk.Frame(parent)
        resource_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        # Systemressourcen-Titel
        title_label = ttk.Label(resource_frame, text="Systemressourcen:", font=("Arial", 12, "bold"))
        title_label.grid(row=0, column=0, padx=5, sticky="w")  # Links ausrichten

        # CPU-Label
        cpu_label = ttk.Label(resource_frame, text="CPU: 0%", font=("Arial", 12))
        cpu_label.grid(row=0, column=1, padx=5)

        # RAM-Label
        ram_label = ttk.Label(resource_frame, text="RAM: 0 GB von 0 GB", font=("Arial", 12))
        ram_label.grid(row=0, column=2, padx=5)

        # GPU-Label
        gpu_label = ttk.Label(resource_frame, text="GPU: Nicht verfügbar", font=("Arial", 12))
        gpu_label.grid(row=0, column=3, padx=5)

         # Platzhalter-Spalte, um die Version nach rechts zu drücken
        resource_frame.grid_columnconfigure(4, weight=1)

        # Version-Titel
        version_label = ttk.Label(resource_frame, text="Version 1.0.0", font=("Arial", 12))
        version_label.grid(row=0, column=5, padx=5, sticky="e")  # Ganz rechts ausrichten

        return cpu_label, ram_label, gpu_label



    # --------------------------------------------------------
    # Beenden
    def quit_application(self):
        self.root.quit()    # Standard-Tkinter-Methode, Haupt-Loop wird verlassen

    # --------------------------------------------------------
    # Chatbot-Steuerung
    def toggle_chatbot(self):
        if self.chatbot_status_var.get() == "Gestartet":
            # Bot ausschalten
            self.chatbot_status_var.set("Gestoppt")

            # Setze alle Chatbot-Status-Labels auf "Aus" in Rot
            self.chatbot_status_label_overview.config(text="Chatbot: Aus", foreground="red")
            self.chatbot_status_label_chat.config(text="Chatbot: Aus", foreground="red")
            self.chatbot_status_label_training.config(text="Chatbot: Aus", foreground="red")

            # entladen
            chatbot_main.config = None
            chatbot_main.faq_data = []
            chatbot_main.model = None
            chatbot_main.tokenizer = None
            chatbot_main.hf_pipeline = None

        else:
            # Bot einschalten und JEDES MAL neu laden
            chatbot_main.init_chatbot()  # Laden von config, model etc.
            self.chatbot_status_var.set("Gestartet")

            # Setze alle Chatbot-Status-Labels auf "An" in Grün
            self.chatbot_status_label_overview.config(text="Chatbot: An", foreground="green")
            self.chatbot_status_label_chat.config(text="Chatbot: An", foreground="green")
            self.chatbot_status_label_training.config(text="Chatbot: An", foreground="green")

    
    def start_chatbot(self):
    # Falls noch nicht gestartet, init_chatbot()
        if chatbot_main.hf_pipeline is None:
            chatbot_main.init_chatbot()
            self.chatbot_status_label.config(text="Chatbot: An", foreground="green")

    def stop_chatbot(self):
        # Bot beenden / entladen
        chatbot_main.config = None
        chatbot_main.faq_data = []
        chatbot_main.model = None
        chatbot_main.tokenizer = None
        chatbot_main.hf_pipeline = None
        self.chatbot_status_label.config(text="Chatbot: Aus", foreground="red")

    def reload_chatbot(self):
        # Lädt IMMER neu
        chatbot_main.config = None
        chatbot_main.faq_data = []
        chatbot_main.model = None
        chatbot_main.tokenizer = None
        chatbot_main.hf_pipeline = None
        chatbot_main.init_chatbot()
        messagebox.showinfo("Info", "Chatbot wurde neu geladen.")

    # --------------------------------------------------------
    # Modelwechsel
    def handle_model_switch(self):
        """Handhabt den Wechsel des Modells."""
        new_model = self.model_var.get()
        message = self.model_manager.switch_model(new_model)
        messagebox.showinfo("Modellwechsel", message)

    # --------------------------------------------------------
    # API-Server
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
                print(f"API ist Offline: {e}")
                self.api_status = "Offline"

        # Server in einem separaten Thread starten
        api_thread = threading.Thread(target=run_server, daemon=True)
        api_thread.start()
        print("API-Server wurde gestartet.")

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
        status = self.api_manager.get_status()  # "Läuft" oder "Gestoppt"
        
        # Farbe nach Status wählen
        if status == "Gestoppt":
            self.api_status_label.config(text=f"API-Status: {status}", foreground="red")
        else:  # Läuft
            self.api_status_label.config(text=f"API-Status: {status}", foreground="green")


    def apply_chat_settings(self):
        """Speichert und übernimmt die Chat-Einstellungen."""
        temperature = float(self.temperature_entry.get())
        self.config_manager.set_param("CHAT", "temperature", temperature)
        messagebox.showinfo("Einstellungen", "Chat-Einstellungen gespeichert.")

    def handle_chat_input(self, event=None):
        if self.chatbot_status_var.get() != "Gestartet":
            messagebox.showwarning("Chatbot inaktiv", "Der Chatbot ist ausgeschaltet!")
            return

        user_input = self.chat_input.get().strip()
        if not user_input:
            return

        self.chat_input.delete(0, tk.END)
        self.append_to_chat("Du", user_input)

        # Chatbot-Antwort abrufen
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
        self.training_status_label.config(text="Training: Läuft...", foreground="green")
        self.trainer.start_training(update_logs_callback=self.training_finished)

    def stop_training(self):
        """Stoppt das Training."""
        self.trainer.stop_training()

    def update_training_logs(self):
        """Lädt die Trainingslogs aus der Datei und zeigt sie im Trainingslog-Widget an."""
        log_path = "training_logs/training_logs.txt"
        try:
            # Prüfen, ob die Datei existiert
            with open(log_path, "r", encoding="utf-8") as log_file:
                logs = log_file.read()
        except FileNotFoundError:
            logs = "Keine Logs verfügbar."
        except Exception as e:
            logs = f"Fehler beim Laden der Logs: {e}"

        # Logs im Text-Widget aktualisieren
        self.training_logs_text.config(state="normal")
        self.training_logs_text.delete("1.0", tk.END)  # Bestehenden Text löschen
        self.training_logs_text.insert(tk.END, logs)   # Neue Logs einfügen
        self.training_logs_text.see("end")             # Automatisch ans Ende scrollen
        self.training_logs_text.config(state="disabled")  # Schreibschutz aktivieren

        # Wiederhole die Aktualisierung alle 30 Sekunden
        self.root.after(30000, self.update_training_logs)


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