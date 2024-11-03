from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from modules.knowledge_base import get_info_by_topic, get_service_description
from modules.recommendation import get_recommendation, get_advanced_recommendation
from utils import format_output, preprocess_text
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from fuzzywuzzy import fuzz, process
import json
import torch
import os

# Pfad zum trainierten Modell
model_path = "./fine_tuned_model"

# Überprüfen, ob das Modell existiert, und ein Standardmodell laden, falls es fehlt
if os.path.exists(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("Verwende das fine-tuned Modell.")
else:
    # Standardmodell von Hugging Face verwenden
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("Verwende das Standardmodell von Hugging Face.")

# Gerät für die Pipeline festlegen (GPU falls verfügbar)
device = 0 if torch.cuda.is_available() else -1
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Wissensdatenbank laden und in das passende Format konvertieren
def load_knowledge_base():
    with open("data/dialogues.json", "r", encoding="utf-8") as file:
        raw_data = json.load(file)
    documents = [
        Document(page_content=item["question"], metadata={"answer": item["answer"]})
        for item in raw_data
    ]
    return documents

# Wissensbasis für RAG vorbereiten
def setup_retriever(knowledge_base):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    faiss_index = FAISS.from_documents(knowledge_base, embeddings)
    retriever = faiss_index.as_retriever()
    return retriever

# Initialisierung der Wissensdatenbank und des Retrievers
knowledge_base = load_knowledge_base()
retriever = setup_retriever(knowledge_base)

# RetrievalQA-Kette für die dynamische Antwortgenerierung
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# Fuzzy Matching für FAQ-Fragen
def load_faq():
    with open("data/dialogues.json", "r", encoding="utf-8") as file:
        return json.load(file)

def get_faq_answer_fuzzy(user_input):
    faq_data = load_faq()
    user_input = preprocess_text(user_input)
    questions = [item["question"] for item in faq_data]
    
    # Fuzzy Matching, um die am besten passende Frage zu finden
    best_match, score = process.extractOne(user_input, questions, scorer=fuzz.token_sort_ratio)
    
    if score > 70:  # Setze eine Schwelle für die Übereinstimmung
        for item in faq_data:
            if item["question"] == best_match:
                return item["answer"]
    return "Ich habe leider keine Antwort auf diese Frage."

# Haupt-Chat-Funktion
def chat():
    print("Starte den Chat (zum Beenden 'exit' eingeben)")
    while True:
        user_input = input("Du: ")
        if user_input.lower() == "exit":
            print("Chat beendet.")
            break

        user_input = preprocess_text(user_input)
        
        # FAQ mit Fuzzy Matching
        response = get_faq_answer_fuzzy(user_input) or qa_chain.run(user_input) or "Entschuldigung, dazu habe ich keine Informationen."

        print(format_output(response))

if __name__ == "__main__":
    chat()
