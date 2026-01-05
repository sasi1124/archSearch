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

---

## Workflows (Diagrams)

### Upload & Approval Workflow

```mermaid
graph TD
  A[User submits artifact<br/>name/description/category/location + images] --> B[Flask Backend /api/artifacts]
  B --> C[SQLite: insert into pending]

  D[Expert reviewer] --> E[UI: Approval Queue]
  E --> F[Flask Backend /api/artifacts/<id>/approve]

  F --> G[SQLite: insert into approved]
  F --> H[Embed text (all-MiniLM-L6-v2)<br/>"name + description"]
  H --> I[FAISS: add vector with approved id<br/>artifact_vectors.index]
  F --> J[SQLite: delete from pending]
```

### Text Search Workflow

```mermaid
graph TD
  A[User enters text query] --> B[UI: Discover]
  B --> C[Flask Backend /api/search/text]
  C --> D[Embed query (all-MiniLM-L6-v2)]
  D --> E[FAISS search<br/>artifact_vectors.index]
  E --> F[Top matching artifact IDs + similarity scores]
  F --> G[SQLite lookup in approved]
  G --> H[Return artifacts + score + image URLs]
  H --> I[UI renders results + confidence + images]
```

### Image Search Workflow

```mermaid
graph TD
  A[User uploads/takes photo] --> B[UI: Discover Image Search]
  B --> C[Flask Backend /api/search/image]
  C --> D[Compute perceptual hash (pHash-like)]
  D --> E[SQLite: scan approved.image_hashes]
  E --> F[Compute hamming distance -> confidence]
  F --> G{Confidence >= threshold?}
  G -- No --> H[Return no matches]
  G -- Yes --> I[Return best match + confidence + image URLs]
  I --> J[UI renders matched artifact]
```

---

## How Embeddings Are Generated (Beginner-Friendly)

Computers can’t directly “understand” text. For semantic search, we convert text into numbers.

### Step 1: Turn text into a vector (a list of numbers)
When an artifact is approved, we build a single text string:

- `"{name} {description}"`

Then we generate an **embedding** using the SentenceTransformers model:

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Output: a **384-dimensional vector** (384 numbers)

This vector acts like a “meaning fingerprint” of the text.

### Step 2: Normalize vectors so cosine similarity works
The backend generates embeddings with normalization enabled:

- `model.encode(..., normalize_embeddings=True)`

Normalization makes vectors unit length, so FAISS inner product behaves like **cosine similarity**.

### Step 3: Store embeddings in FAISS
The vector is inserted into FAISS (`artifact_vectors.index`) with the SQLite `approved.id` as its ID.

### Step 4: Searching is the same process
When you type a query:
1. The query text is embedded into another 384-number vector
2. FAISS finds the closest vectors already stored
3. The backend uses those IDs to fetch full artifact details from SQLite

### Quick analogy (RGB colors)
Think of how a color can be represented as numbers like `[R, G, B]`.

Embeddings do the same idea for **meaning**, except they use **384 numbers** instead of 3.
Texts with similar meaning produce vectors that are “close” to each other.

---

## Docs

### Common tasks

#### Add an artifact that is searchable by text and image
1. Upload an artifact with **name/description** and at least **one image** (Upload tab)
2. Approve it (Approval Queue tab)
3. Discover:
   - Text search uses FAISS + embeddings
   - Image search uses perceptual hash matching against `approved.image_hashes`

#### Verify an artifact has images stored
- In SQLite, `approved.images` should contain one or more filenames.

#### Verify an artifact has image hashes stored
- In SQLite, `approved.image_hashes` should contain one or more hex hashes.

#### Rebuild / reset demo data
If you want a clean slate, delete:
- `backend/archsearch.db`
- `backend/artifact_vectors.index`
- files under `backend/static/images/`

Then restart the server (`init_db()` will recreate the schema).

---

## Beginner Q&A (FAQ)

### Q: Is this using an LLM like ChatGPT?
**A:** No. This project uses an **embedding model** (SentenceTransformers) for semantic search, not a chat/completions LLM. The model is `sentence-transformers/all-MiniLM-L6-v2`.

### Q: What’s an embedding?
**A:** An embedding is a numeric representation of text (here: 384 numbers) that captures the *meaning* of the text. Similar meanings → vectors close together.

### Q: Why do we need FAISS?
**A:** FAISS is a fast library for searching through many vectors. It lets us quickly find the most semantically similar artifacts.

### Q: Where is the “source of truth” for artifact data?
**A:** SQLite (`backend/archsearch.db`) is the source of truth for artifact fields. FAISS stores only vectors + IDs.

### Q: When does the FAISS index update?
**A:** When an artifact is **approved**, the backend creates an embedding and adds it to `artifact_vectors.index` with the approved artifact’s `id`.

### Q: Why doesn’t image search return results for some artifacts?
**A:** Image search only works for artifacts that were submitted with images and then approved. Those records must have `approved.images` and `approved.image_hashes` populated.

### Q: Why do I get a “403” when using localhost?
**A:** On some macOS setups, system services intercept port 5000 and respond with `Server: AirTunes/...`. Use `http://127.0.0.1:5050/`.

### Q: Where are uploaded images stored?
**A:** On disk at `backend/static/images/`. SQLite stores the filenames in `images`.

### Q: How do I inspect what’s inside the database?
**A:**
```bash
sqlite3 backend/archsearch.db
```
Then run:
```sql
SELECT * FROM approved;
SELECT * FROM pending;
```

### Q: How do I inspect the FAISS index?
**A:**
```bash
.venv/bin/python backend/inspect_index.py
```

### Q: I approved an artifact but text search doesn’t find it.
**A:** Ensure:
- the approval request succeeded (no server errors)
- `artifact_vectors.index` exists and has vectors
- you’re searching with enough characters (very short queries are ignored)

### Q: How is “confidence” computed for text search?
**A:** Text search returns a cosine similarity score. The UI maps it to 0–100 for display and filters low-confidence results.

### Q: How is “confidence” computed for image search?
**A:** It’s based on the Hamming distance between perceptual hashes, mapped to a 0–100 scale.
