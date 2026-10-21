import time
from sentence_transformers import SentenceTransformer

start = time.time()
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
print("Modell geladen in", round(time.time() - start, 1), "Sekunden")

frage = ["How does the scheduler pick the next task to run?"]
texte = [
    "EEVDF selects the eligible task with the earliest virtual deadline.",
    "The network driver initializes the transmit queue.",
]

frage_vektor = model.encode(frage, prompt_name="query")
text_vektoren = model.encode(texte)

print("Vektorgröße:", frage_vektor.shape[1])
print("Ähnlichkeit zu den beiden Texten:", model.similarity(frage_vektor, text_vektoren))