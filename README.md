# BA1 RAG – Vektordatenbank (Teil Otto)

Teil der Bachelorarbeit 1 „Retrieval-Augmented Generation für technische Dokumentation und Softwareartefakte“ (Otto Detki, Michael Öschlberger; Betreuer: Simon Sigl).

Dieses Projekt baut aus dem Linux-Kernel (derzeit nur der Scheduler) eine Vektordatenbank auf und stellt eine Suchfunktion bereit: **Frage rein, passendste Textstücke mit Quelldatei raus.** Diese Treffer gibt Michaels Teil an das Sprachmodell weiter.

**Stand (22.10.2026):** Durchstich funktioniert mit dem Scheduler-Ausschnitt von Kernel 6.18 (1.174 Chunks).

---

## Die Programme

| Datei | Was sie macht |
|---|---|
| `config.py` | Alle Einstellungen an einer Stelle: Kernel-Version, Modell, Chunk-Größe, Distanzmaß. Wer etwas ändern will, ändert nur hier. |
| `build_index.py` | Baut die Datenbank: Kernel-Dateien lesen → in Stücke (Chunks) schneiden → in Vektoren umwandeln → in ChromaDB speichern → Messwerte in `messungen.csv` schreiben. |
| `query.py` | Die Suche und **die Schnittstelle zu Michaels Teil**: Funktion `suche(frage, k)`. |
| `test_embedding.py` | Kleiner Test, ob das Embedding-Modell funktioniert. |
| `messungen.csv` | Automatisch erzeugtes Protokoll jedes Indexierungslaufs (Dateien, Chunks, Laufzeit, Hardware, Kernel-Commit). |
| `requirements.txt` | Exakte Paketversionen. |

Der Ordner `data/` (Kernel-Quellcode und Datenbank) ist **nicht** im Git, weil er groß ist und jederzeit neu erzeugt werden kann.

---

## Schnittstelle für Michael

```python
from query import suche

treffer = suche("How does EEVDF pick the next task to run?", k=5)
```

Rückgabe: eine Liste, sortiert nach Ähnlichkeit, z. B.

```python
{"rang": 1,
 "pfad": "Documentation/scheduler/sched-eevdf.rst",
 "typ": "doku",            # oder "code"
 "chunk_nr": 0,
 "aehnlichkeit": 0.7337,   # Kosinus-Ähnlichkeit, höher = passender
 "text": "..."}            # das eigentliche Textstück für das LLM
```

Das Modell wird beim ersten Aufruf einmal geladen (einige Sekunden), danach dauert jede Suche unter einer Sekunde.

---

## Einrichtung (Windows, PowerShell)

```powershell
git clone https://its-git.fh-salzburg.ac.at/fhs51270/ba1-rag.git
cd ba1-rag
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Mit NVIDIA-Grafikkarte:** `requirements.txt` enthält die PyTorch-Version für den Prozessor. Für die GPU danach PyTorch mit CUDA nachinstallieren, den passenden Befehl liefert https://pytorch.org/get-started (die RTX 5080 braucht eine aktuelle CUDA-Version), z. B.:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall
```

Die Programme erkennen die Grafikkarte automatisch.

### Kernel-Ausschnitt holen

```powershell
mkdir data
cd data
git clone --depth 1 --branch v6.18 --filter=blob:none --sparse https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git linux
cd linux
git sparse-checkout set kernel/sched Documentation/scheduler
cd ..\..
```

Bei einer anderen Kernel-Version `v6.18` hier und in `config.py` ersetzen.

### Datenbank aufbauen und abfragen

```powershell
python build_index.py --test     # nur 3 Dateien, ca. 30 s
python build_index.py            # vollständig
python query.py "How does EEVDF pick the next task to run?"
```

Beim ersten Start wird das Embedding-Modell (ca. 1,2 GB) heruntergeladen.

**Laufzeit vollständiger Scheduler-Ausschnitt:** ca. 107 Minuten auf einem Notebook-Prozessor (i7-8665U). Auf einer Grafikkarte deutlich schneller.

---

## Einstellungen und Entscheidungen

| Einstellung | Wert | Kurzbegründung |
|---|---|---|
| Vektordatenbank | ChromaDB 1.5.9 | läuft direkt in Python, ohne Server und ohne Docker; HNSW-Index |
| Embedding-Modell | Qwen3-Embedding-0.6B | auf Code-Retrieval ausgelegt; klein genug, um neben dem LLM auf 16 GB VRAM zu laufen |
| Distanzmaß | Kosinus | Standard im Vektorraummodell, unabhängig von der Textlänge; Chroma-Standard L2 wird überschrieben |
| Chunking | 512 Tokens, 64 Überlappung | Startwert für den Durchstich; Top-5 ≈ 2.500 Tokens Kontext für das LLM |
| Python | 3.13 | von ChromaDB und PyTorch unterstützt, keine Kompatibilitätsprobleme |
| Kernel | v6.18 (vorläufig) | Commit `7d0a66e4bb9081d75c82ec4957c50034cb0ea449` |

Fragen werden mit der Anweisung `prompt_name="query"` eingebettet, Dokumente ohne. Auf der CPU wird in float32 gerechnet, auf der GPU in float16.

---

## Erste Beobachtungen

- Fragen in natürlicher Sprache (EEVDF, Deadline, Echtzeit) liefern 3–4 von 5 relevanten Treffern.
- Fragen nach exakten Funktionsnamen (z. B. `update_curr`) funktionieren schlecht → Motivation für hybrides Retrieval mit BM25.
- Inhaltsverzeichnis-Chunks landen teils weit vorne → Ansatzpunkt für strukturbewusstes Chunking.

## Hinweis PowerShell

In Dateien nie mit `>` schreiben (erzeugt UTF-16), sondern z. B. `pip freeze | Set-Content -Encoding ascii requirements.txt`.
