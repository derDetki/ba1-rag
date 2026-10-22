"""Abfrage der Vektordatenbank: Frage rein, passendste Chunks raus.

Das ist die Schnittstelle zu Michaels Teil. Sein Code ruft nur die
Funktion suche() auf und bekommt eine Liste von Treffern zurueck.

Aufruf zum Ausprobieren:
    python query.py "How does EEVDF pick the next task?"
    python query.py "How does EEVDF pick the next task?" --k 3
"""

import argparse

import chromadb
import torch
from sentence_transformers import SentenceTransformer

import config

_modell = None
_collection = None


def _laden():
    """Laedt Modell und Datenbank nur einmal, auch bei vielen Fragen."""
    global _modell, _collection
    if _modell is None:
        geraet = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if geraet == "cuda" else torch.float32
        _modell = SentenceTransformer(
            config.MODELL_NAME, device=geraet, model_kwargs={"dtype": dtype}
        )
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        _collection = client.get_collection(config.COLLECTION_NAME)
    return _modell, _collection


def suche(frage, k=5):
    """Sucht die k passendsten Chunks zu einer Frage.

    Rueckgabe: Liste von Dictionaries, sortiert nach Aehnlichkeit, z. B.
        {"rang": 1, "pfad": "kernel/sched/fair.c", "typ": "code",
         "chunk_nr": 12, "aehnlichkeit": 0.61, "text": "..."}
    """
    modell, collection = _laden()
    vektor = modell.encode(
        [frage], prompt_name="query", normalize_embeddings=True
    )
    ergebnis = collection.query(
        query_embeddings=vektor.tolist(),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    treffer = []
    for rang, (text, meta, distanz) in enumerate(
        zip(ergebnis["documents"][0], ergebnis["metadatas"][0], ergebnis["distances"][0]),
        start=1,
    ):
        treffer.append({
            "rang": rang,
            "pfad": meta["pfad"],
            "typ": meta["typ"],
            "chunk_nr": meta["chunk_nr"],
            # Chroma liefert bei Kosinus die Distanz (1 - Aehnlichkeit)
            "aehnlichkeit": round(1 - distanz, 4),
            "text": text,
        })
    return treffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frage")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    for t in suche(args.frage, args.k):
        print(f"\n#{t['rang']}  {t['pfad']}  (Chunk {t['chunk_nr']}, {t['typ']}, "
              f"Aehnlichkeit {t['aehnlichkeit']})")
        vorschau = t["text"][:300].replace("\n", " ")
        print(f"    {vorschau} ...")


if __name__ == "__main__":
    main()
