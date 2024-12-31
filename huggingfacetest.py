from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
result = generator("Was ist KI?", max_length=50)
print(result)
