"""Baut die Vektordatenbank aus dem Kernel-Ausschnitt auf.

Ablauf: Dateien lesen -> in feste Token-Chunks schneiden -> einbetten
-> in ChromaDB speichern -> Messwerte in messungen.csv festhalten.

Aufruf:
    python build_index.py          # kompletter Ausschnitt
    python build_index.py --test   # nur 3 Dateien, zum schnellen Ausprobieren
"""

import argparse
import csv
import platform
import subprocess
import time
from datetime import datetime

import chromadb
import torch
from sentence_transformers import SentenceTransformer

import config


def dateien_sammeln():
    """Alle relevanten Dateien aus den Quellordnern, sortiert fuer Reproduzierbarkeit."""
    dateien = []
    for ordner in config.QUELL_ORDNER:
        for pfad in sorted((config.LINUX_DIR / ordner).rglob("*")):
            if pfad.is_file() and pfad.suffix in config.DATEIENDUNGEN:
                dateien.append(pfad)
    return dateien


def in_chunks_schneiden(text, tokenizer):
    """Schneidet einen Text in Chunks fester Token-Laenge mit Ueberlappung."""
    token_ids = tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"]
    schritt = config.CHUNK_GROESSE - config.CHUNK_UEBERLAPPUNG
    chunks = []
    for start in range(0, len(token_ids), schritt):
        stueck = token_ids[start:start + config.CHUNK_GROESSE]
        chunks.append((start, tokenizer.decode(stueck)))
        if start + config.CHUNK_GROESSE >= len(token_ids):
            break
    return chunks, len(token_ids)


def kernel_commit():
    """Commit-Hash des ausgecheckten Kernel-Stands (fuer die Methodik)."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(config.LINUX_DIR), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unbekannt"


def messung_speichern(werte):
    neu = not config.MESSUNGEN_DATEI.exists()
    with open(config.MESSUNGEN_DATEI, "a", newline="", encoding="utf-8") as f:
        schreiber = csv.DictWriter(f, fieldnames=list(werte.keys()))
        if neu:
            schreiber.writeheader()
        schreiber.writerow(werte)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="nur 3 Dateien verarbeiten")
    args = parser.parse_args()

    t_gesamt = time.time()

    # 1. Dateien sammeln
    dateien = dateien_sammeln()
    if args.test:
        dateien = dateien[:3]
    print(f"{len(dateien)} Dateien gefunden")

    # 2. Modell laden (auf der CPU in float32, siehe Protokoll)
    geraet = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if geraet == "cuda" else torch.float32
    modell = SentenceTransformer(
        config.MODELL_NAME, device=geraet, model_kwargs={"dtype": dtype}
    )
    tokenizer = modell.tokenizer
    print(f"Modell geladen ({geraet}, {dtype})")

    # 3. Chunks bilden
    ids, texte, metadaten = [], [], []
    token_gesamt = 0
    for pfad in dateien:
        rel = pfad.relative_to(config.LINUX_DIR).as_posix()
        text = pfad.read_text(encoding="utf-8", errors="replace")
        chunks, n_tokens = in_chunks_schneiden(text, tokenizer)
        token_gesamt += n_tokens
        for nr, (start, chunk_text) in enumerate(chunks):
            ids.append(f"{rel}::{nr}")
            texte.append(chunk_text)
            metadaten.append({
                "pfad": rel,
                "typ": "doku" if rel.startswith("Documentation/") else "code",
                "chunk_nr": nr,
                "start_token": start,
                "kernel_version": config.KERNEL_VERSION,
            })
    print(f"{len(texte)} Chunks aus {token_gesamt} Tokens")

    # 4. Einbetten
    t_embed = time.time()
    vektoren = modell.encode(
        texte,
        batch_size=config.BATCH_GROESSE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    dauer_embed = time.time() - t_embed
    print(f"Einbetten: {dauer_embed:.1f} s")

    # 5. In ChromaDB speichern (Collection wird jedes Mal neu aufgebaut)
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        configuration={"hnsw": {"space": config.DISTANZMASS}},
        embedding_function=None,
    )
    max_batch = client.get_max_batch_size()
    for i in range(0, len(ids), max_batch):
        collection.add(
            ids=ids[i:i + max_batch],
            documents=texte[i:i + max_batch],
            embeddings=vektoren[i:i + max_batch].tolist(),
            metadatas=metadaten[i:i + max_batch],
        )
    print(f"{collection.count()} Eintraege in der Datenbank")

    dauer_gesamt = time.time() - t_gesamt
    print(f"Gesamtdauer: {dauer_gesamt:.1f} s")

    # 6. Messwerte festhalten
    messung_speichern({
        "datum": datetime.now().isoformat(timespec="seconds"),
        "testlauf": args.test,
        "kernel_version": config.KERNEL_VERSION,
        "kernel_commit": kernel_commit(),
        "dateien": len(dateien),
        "tokens": token_gesamt,
        "chunks": len(texte),
        "chunk_groesse": config.CHUNK_GROESSE,
        "ueberlappung": config.CHUNK_UEBERLAPPUNG,
        "modell": config.MODELL_NAME,
        "geraet": geraet,
        "dtype": str(dtype),
        "prozessor": platform.processor(),
        "dauer_embed_s": round(dauer_embed, 1),
        "dauer_gesamt_s": round(dauer_gesamt, 1),
    })
    print("Messung in messungen.csv gespeichert")


if __name__ == "__main__":
    main()
