import tkinter as tk
from tkinter import ttk, messagebox
from train_model import main as start_training_process  # Importiere die main-Funktion aus train_model.py
from api_main import allow_origins, allow_credentials, allow_methods, allow_headers, app  # Importiere die FastAPI-App und Parameter
from chatbot_main import get_faq_answer_fuzzy  # Importiere die benötigten Funktionen
from utils import preprocess_text
import os
import threading
import uvicorn


class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Management Dashboard")
        self.root.geometry("800x600")

        # Globale Variable für Logs definieren
        self.chat_logs_dir = "chat_logs"
        os.makedirs(self.chat_logs_dir, exist_ok=True)

        # API-Status
        self.api_status = "Offline"

        # API starten
        self.start_api_server()

        # Widgets erstellen
        self.create_widgets()

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

    # Tab: Übersicht
    def create_overview_tab(self):
        ttk.Label(self.overview_tab, text="CPU- und RAM-Auslastung", font=("Arial", 16)).pack(pady=10)
        self.cpu_label = ttk.Label(self.overview_tab, text="CPU: 0%", font=("Arial", 12))
        self.cpu_label.pack(pady=5)
        self.ram_label = ttk.Label(self.overview_tab, text="RAM: 0 GB von 0 GB", font=("Arial", 12))
        self.ram_label.pack(pady=5)

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

        self.quit_button = ttk.Button(self.overview_tab, text="Beenden", command=self.root.quit)
        self.quit_button.pack(pady=10)

    # Tab: Training
    def create_training_tab(self):
        ttk.Label(self.training_tab, text="Training Parameter", font=("Arial", 16)).pack(pady=10)

        ttk.Label(self.training_tab, text="Epochen:").pack(pady=5)
        self.epochs_entry = ttk.Entry(self.training_tab)
        self.epochs_entry.pack(pady=5)

        ttk.Label(self.training_tab, text="Lernrate:").pack(pady=5)
        self.lr_entry = ttk.Entry(self.training_tab)
        self.lr_entry.pack(pady=5)

        ttk.Label(self.training_tab, text="Batchgröße:").pack(pady=5)
        self.batch_entry = ttk.Entry(self.training_tab)
        self.batch_entry.pack(pady=5)

        self.train_button = ttk.Button(self.training_tab, text="Training starten", command=self.start_training)
        self.train_button.pack(pady=20)

        self.training_progress = ttk.Label(self.training_tab, text="Fortschritt: Noch nicht gestartet", font=("Arial", 12))
        self.training_progress.pack(pady=10)

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

    # Tab: Einstellung
    def create_settings_tab(self):
        ttk.Label(self.settings_tab, text="API Einstellungen", font=("Arial", 16)).pack(pady=10)

        ttk.Label(self.settings_tab, text="Erlaubte Domains (allow_origins):").pack(pady=5)
        self.origins_entry = ttk.Entry(self.settings_tab)
        self.origins_entry.insert(0, ", ".join(allow_origins))
        self.origins_entry.pack(pady=5)

        ttk.Label(self.settings_tab, text="Credentials erlauben (allow_credentials):").pack(pady=5)
        self.credentials_entry = ttk.Entry(self.settings_tab)
        self.credentials_entry.insert(0, str(allow_credentials))
        self.credentials_entry.pack(pady=5)

        ttk.Label(self.settings_tab, text="Erlaubte Methoden (allow_methods):").pack(pady=5)
        self.methods_entry = ttk.Entry(self.settings_tab)
        self.methods_entry.insert(0, ", ".join(allow_methods))
        self.methods_entry.pack(pady=5)

        ttk.Label(self.settings_tab, text="Erlaubte Header (allow_headers):").pack(pady=5)
        self.headers_entry = ttk.Entry(self.settings_tab)
        self.headers_entry.insert(0, ", ".join(allow_headers))
        self.headers_entry.pack(pady=5)

    def start_training(self):
        epochs = self.epochs_entry.get()
        lr = self.lr_entry.get()
        batch_size = self.batch_entry.get()

        if not epochs or not lr or not batch_size:
            messagebox.showerror("Fehler", "Bitte alle Parameter eingeben!")
            return

        try:
            epochs = int(epochs)
            lr = float(lr)
            batch_size = int(batch_size)

            self.training_progress.config(text="Training läuft...")
            start_training_process()  # Starte das Training
            self.training_progress.config(text="Training abgeschlossen!")
        except Exception as e:
            messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten: {str(e)}")


# Anwendung starten
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
