@ -201,3 +201,176 @@ Backend dependencies come from `backend/requirements.txt`:

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