# ArchSearch: A Smart Library for Archaeologists

ArchSearch is a small Flask + SQLite prototype for submitting, reviewing, and searching archaeological artifacts.

It supports:
- **Text semantic search** (FAISS vector index + sentence-transformers embeddings)
- **Image similarity search** (perceptual hash / pHash-style matching)
- **Upload → Approval → Publish** workflow
- Simple **autocomplete** suggestions

---

## Quick Start (macOS)

### 1) Create/activate the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3) Run the server

```bash
.venv/bin/python backend/app.py
```

Open:
- `http://127.0.0.1:5050/`

> Note: on some macOS setups, **port 5000** may be intercepted by system services and return a `403` for `http://localhost:5000`. This project is configured to run on **5050** to avoid that.

---

## Project Layout

```
archSearch/
  README.md
  backend/
    app.py                 # Flask backend (API + template)
    templates/
      archSearch.html       # Single-page UI
    static/
      images/               # Uploaded artifact images
    archsearch.db           # SQLite database
    artifact_vectors.index  # FAISS text vector index
    inspect_index.py        # Helper to inspect FAISS index stats
```

---

## Core Components

### Backend (Flask)
- Main server: `backend/app.py`
- Serves:
  - UI: `GET /` → `backend/templates/archSearch.html`
  - API: `GET/POST /api/...`

### Database (SQLite)
- File: `backend/archsearch.db`
- Tables:
  - `pending`: newly submitted artifacts awaiting expert review
  - `approved`: verified artifacts searchable by the public

Each artifact stores:
- `name`, `description`, `category`, `location`, `uploadedBy`, `license`, `date`
- `images`: comma-separated list of image filenames saved under `backend/static/images/`
- `image_hashes`: comma-separated list of perceptual hashes for image search

> `init_db()` also runs lightweight migrations to add missing columns when you upgrade the code.

---

## Text Search (Semantic Search)

### “LLM model” / embedding model used
This project does **semantic embeddings** using **SentenceTransformers**:
- Model: **`sentence-transformers/all-MiniLM-L6-v2`**
- Embedding dimension: **384**

This is not a chat/completion LLM; it’s an **embedding model** used for vector search.

### How FAISS is used
- Index file: `backend/artifact_vectors.index`
- Index type: cosine similarity implemented as inner-product on normalized vectors (FAISS `IndexFlatIP` wrapped in `IndexIDMap2`)

#### When artifacts are indexed
- On **approval**, the backend:
  1) inserts the artifact into `approved`
  2) creates a text embedding from `"{name} {description}"`
  3) adds it to FAISS keyed by the **approved row ID**
  4) deletes the original row from `pending`

#### How text search works
- Endpoint: `POST /api/search/text`
- Steps:
  1) embed user query
  2) search FAISS for nearest IDs
  3) fetch full rows from SQLite `approved`
  4) return results including a similarity `score`

#### Confidence filtering
The UI shows a human-friendly “confidence” derived from the cosine similarity score.
Results are filtered so only matches meeting the configured minimum threshold are returned.

---

## Image Search (Visual Similarity)

Image search is implemented using a lightweight **perceptual hash** (pHash-like) approach.

- Endpoint: `POST /api/search/image` (multipart form-data, key: `file`)
- On upload, the backend computes a hash for the query image.
- It compares that hash against `approved.image_hashes` to find visually similar images.

Current behavior:
- returns the **single best match** only if it meets a minimum confidence threshold
- returns:
  - `name`, `description`, `category`, `location`
  - `confidence` (0–100)
  - `images` (URLs) for display

### Important: why image search may return no results
Only artifacts that were **submitted with images** (and then **approved**) will have `images` and `image_hashes` populated.
Older records may have empty columns.

---

## API Summary

- `GET /` → UI
- `POST /api/artifacts` → submit artifact (multipart form-data; supports images)
- `GET /api/artifacts/pending` → pending list (review)
- `POST /api/artifacts/<id>/approve` → approve + index in FAISS
- `POST /api/artifacts/<id>/reject` → reject with reason
- `GET /api/artifacts/mine?user=<name>` → user submissions

Search:
- `POST /api/search/text` → semantic text search
- `GET /api/search/suggest?q=<prefix>&limit=10` → autocomplete
- `POST /api/search/image` → image similarity search

---

## Inspecting Storage Artifacts

### Inspect SQLite

```bash
sqlite3 backend/archsearch.db
```

Example:
```sql
SELECT id, name, images, image_hashes FROM approved;
```

### Inspect FAISS index

```bash
.venv/bin/python backend/inspect_index.py
```

This prints basic information (vector count, dimension, etc.).

---

## Clearing Data (Reset)

If you want a clean slate:

1) Stop the server
2) Remove runtime artifacts:
- `backend/archsearch.db`
- `backend/artifact_vectors.index`
- `backend/static/images/*`

Then restart the server so it recreates the DB.

---

## Dependencies

Backend dependencies come from `backend/requirements.txt`:
- `flask`, `flask-cors`
- `faiss-cpu`
- `sentence-transformers`, `torch`
- `numpy`, `Pillow`

---

## Notes / Troubleshooting

- **macOS localhost 403**: if `http://localhost:5000` returns 403 with `Server: AirTunes/...`, use `http://127.0.0.1:5050/`.
- **First run downloads model**: SentenceTransformers downloads `all-MiniLM-L6-v2` on first run and caches it under your user cache (typically `~/.cache`).
