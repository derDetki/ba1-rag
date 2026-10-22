"""Zentrale Einstellungen fuer die Vektordatenbank (BA1 RAG, Teil Otto).

Alle Werte, die sich im Laufe des Projekts aendern koennen, stehen hier,
damit der restliche Code unveraendert bleibt.
"""

from pathlib import Path

# --- Pfade -------------------------------------------------------------------
PROJEKT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJEKT_DIR / "data"
LINUX_DIR = DATA_DIR / "linux"          # geklonter Kernel-Ausschnitt
DB_DIR = DATA_DIR / "chroma"            # Speicherort der ChromaDB
MESSUNGEN_DATEI = PROJEKT_DIR / "messungen.csv"

# --- Korpus ------------------------------------------------------------------
KERNEL_VERSION = "v6.18"                # vorlaeufig, mit Michael abzustimmen
QUELL_ORDNER = [
    "kernel/sched",
    "Documentation/scheduler",
]
DATEIENDUNGEN = {".c", ".h", ".rst", ".txt"}

# --- Embedding-Modell --------------------------------------------------------
MODELL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_GROESSE = 8                       # Chunks pro Rechenschritt

# --- Chunking (fester Durchstich) --------------------------------------------
CHUNK_GROESSE = 512                     # Tokens pro Chunk
CHUNK_UEBERLAPPUNG = 64                 # Tokens Ueberlappung

# --- Datenbank ---------------------------------------------------------------
COLLECTION_NAME = "kernel_sched"
DISTANZMASS = "cosine"                  # Chroma-Standard waere "l2"
