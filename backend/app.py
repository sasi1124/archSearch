import os
import sqlite3
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import faiss
import numpy as np
from werkzeug.utils import secure_filename

# NEW: real embeddings
from sentence_transformers import SentenceTransformer

# --- Config ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'archsearch.db')

# --- Flask Setup ---
app = Flask(__name__)
# For a local prototype (including running the frontend via file:// which uses Origin: null),
# allow all origins on API routes.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- SQLite Setup ---
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    # Help avoid "database is locked" for concurrent reads/writes in dev
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    # Pending artifacts
    c.execute('''CREATE TABLE IF NOT EXISTS pending (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        category TEXT,
        location TEXT,
        uploadedBy TEXT,
        license TEXT,
        date TEXT,
        images TEXT,
        status TEXT DEFAULT 'pending',
        rejectionReason TEXT
    )''')
    # Approved artifacts
    c.execute('''CREATE TABLE IF NOT EXISTS approved (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        description TEXT,
        category TEXT,
        location TEXT,
        uploadedBy TEXT,
        license TEXT,
        date TEXT,
        images TEXT,
        citation TEXT
    )''')
    conn.commit()
    conn.close()

# --- FAISS Setup ---
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'artifact_vectors.index')

# Use a real embedding model
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        # This will download the model on first run (cached under ~/.cache/torch/...) 
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_embedding_dim() -> int:
    # Model provides the true dimension (should be 384 for MiniLM-L6-v2)
    return int(get_embedding_model().get_sentence_embedding_dimension())


EMBEDDING_DIM = 384  # kept for readability; validated at runtime
# Minimum cosine similarity required to consider any match.
MIN_SEARCH_SCORE = 0.35
# Only return results that meet the UI "confidence" requirement.
# 75% confidence => score >= 0.5 using pct=((score+1)/2)*100
MIN_RETURN_SCORE = 0.5
# If the best match is not clearly better than the next one, treat it as noise.
MIN_SCORE_GAP = 0.08
# Basic guardrail: don't attempt semantic search for very short queries.
MIN_QUERY_CHARS = 3

def _new_index_with_ids():
    """Create an index that supports storing explicit IDs.

    Use cosine similarity via inner product on L2-normalized vectors.
    """
    dim = get_embedding_dim()
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap2(base)


def get_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        idx = faiss.read_index(FAISS_INDEX_PATH)
        # If an old L2 index exists on disk, upgrade it.
        if isinstance(idx, faiss.IndexFlatL2):
            idx = faiss.IndexIDMap2(faiss.IndexFlatIP(get_embedding_dim()))
        return idx
    else:
        return _new_index_with_ids()

def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)

# --- Embedding ---

def embed_text(text: str) -> np.ndarray:
    """Real semantic embedding using sentence-transformers.

    Returns a float32 vector.
    """
    text = (text or '').strip()
    if not text:
        return np.zeros((get_embedding_dim(),), dtype=np.float32)

    model = get_embedding_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)

# --- Tokenization and Overlap Functions ---
def _tokenize(text: str):
    import re
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


def _token_overlap(a: str, b: str) -> int:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    return len(sa.intersection(sb))

# --- API Endpoints ---
@app.route('/api/artifacts', methods=['POST'])
def submit_artifact():
    # Accept multipart/form-data for image upload
    name = request.form.get('name')
    description = request.form.get('description')
    category = request.form.get('category')
    location = request.form.get('location')
    uploadedBy = request.form.get('uploadedBy')
    license = request.form.get('license')
    date = request.form.get('date')
    # Handle images (save filenames for now)
    image_filenames = []
    if 'images' in request.files:
        images = request.files.getlist('images')
        upload_folder = os.path.join(os.path.dirname(__file__), 'static', 'images')
        os.makedirs(upload_folder, exist_ok=True)  # Ensure the upload folder exists

        for img in images:
            if img and img.filename:
                filename = secure_filename(img.filename)
                save_path = os.path.join(upload_folder, filename)
                img.save(save_path)
                image_filenames.append(filename)

    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO pending (name, description, category, location, uploadedBy, license, date, images)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (name, description, category, location, uploadedBy, license, date, ','.join(image_filenames)))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'images': image_filenames})

@app.route('/api/artifacts/pending', methods=['GET'])
def get_pending():
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM pending')
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route('/api/artifacts/approved', methods=['GET'])
def get_approved():
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM approved')
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(rows)

@app.route('/api/artifacts/<int:artifact_id>/approve', methods=['POST'])
def approve_artifact(artifact_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM pending WHERE id=?', (artifact_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Not found'}), 404

    # Move to approved table and get the new ID
    c.execute('''INSERT INTO approved (name, description, category, location, uploadedBy, license, date, images, citation)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (row['name'], row['description'], row['category'], row['location'], row['uploadedBy'], row['license'], row['date'], row['images'], 'Verified by Expert Panel'))
    new_id = c.lastrowid

    # Add to FAISS index (and persist). This is an incremental update so the index
    # stays current immediately after approval.
    index = get_faiss_index()
    if not isinstance(index, faiss.IndexIDMap) and not isinstance(index, faiss.IndexIDMap2):
        index = faiss.IndexIDMap2(index)

    text_to_embed = f"{row['name']} {row['description']}"
    embedding = embed_text(text_to_embed)

    index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([new_id], dtype=np.int64))
    save_faiss_index(index)

    # Delete from pending
    c.execute('DELETE FROM pending WHERE id=?', (artifact_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'new_id': new_id})

@app.route('/api/artifacts/<int:artifact_id>/reject', methods=['POST'])
def reject_artifact(artifact_id):
    data = request.json
    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE pending SET status=?, rejectionReason=? WHERE id=?', ('rejected', data.get('reason', ''), artifact_id))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/search/text', methods=['POST'])
def search_text():
    data = request.json or {}
    query = (data.get('query') or '').strip()
    k = data.get('k', 5)  # Number of results to return

    # Optional: allow client to request a minimum confidence (0..100)
    min_conf = data.get('minConfidence')
    if isinstance(min_conf, (int, float)):
        min_conf = float(min_conf)
        min_conf = max(0.0, min(100.0, min_conf))
        min_return_score = (min_conf / 50.0) - 1.0
    else:
        min_return_score = MIN_RETURN_SCORE

    if not query:
        return jsonify({'results': []})

    # Exact match pass (case-insensitive) so short/new artifacts are searchable
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM approved WHERE lower(name)=lower(?) ORDER BY id DESC LIMIT 1', (query,))
    exact = c.fetchone()
    conn.close()
    if exact:
        art = dict(exact)
        # Convert stored filenames into URLs for the frontend
        imgs = (art.get('images') or '').split(',') if art.get('images') else []
        art['images'] = [f"/static/images/{i.strip()}" for i in imgs if i.strip()]
        art['score'] = 1.0
        return jsonify({'results': [art]})

    # Guardrail
    if len(query) < MIN_QUERY_CHARS:
        return jsonify({'results': []})

    # Load index
    index = get_faiss_index()
    if index.ntotal == 0:
        return jsonify({'results': []})

    query_embedding = embed_text(query)
    scores, ids = index.search(np.array([query_embedding], dtype=np.float32), k)

    best_score = float(scores[0][0]) if scores.size else -1.0
    if best_score < MIN_SEARCH_SCORE:
        return jsonify({'results': []})

    if scores.shape[1] >= 2:
        second_score = float(scores[0][1])
        if (best_score - second_score) < MIN_SCORE_GAP:
            return jsonify({'results': []})

    conn = get_db()
    c = conn.cursor()

    # Only keep results at/above the requested confidence threshold
    found_pairs = []
    for j in range(ids.shape[1]):
        art_id = int(ids[0][j])
        score = float(scores[0][j])
        if art_id >= 0 and score >= min_return_score:
            found_pairs.append((art_id, score))

    found_ids = [p[0] for p in found_pairs]
    if not found_ids:
        conn.close()
        return jsonify({'results': []})

    placeholders = ','.join('?' for _ in found_ids)
    c.execute(f'SELECT * FROM approved WHERE id IN ({placeholders})', found_ids)
    artifacts = [dict(row) for row in c.fetchall()]
    conn.close()

    id_to_artifact = {art['id']: art for art in artifacts}

    # Attach similarity score to each returned artifact
    results = []
    for art_id, score in found_pairs:
        art = id_to_artifact.get(art_id)
        if not art:
            continue
        art = dict(art)
        imgs = (art.get('images') or '').split(',') if art.get('images') else []
        art['images'] = [f"/static/images/{i.strip()}" for i in imgs if i.strip()]
        art['score'] = score
        results.append(art)

    return jsonify({'results': results})

@app.route('/api/search/image', methods=['POST'])
def search_image():
    # Stub: always returns empty
    return jsonify({'analysis': 'Image search is not implemented in this prototype.', 'matches': []})

@app.route('/api/artifacts/bulk', methods=['POST'])
def bulk_upload():
    data = request.json
    artifacts = data.get('artifacts', [])
    uploadedBy = data.get('uploadedBy', 'Excel Import')
    license = data.get('license', '')
    conn = get_db()
    c = conn.cursor()
    for artifact in artifacts:
        c.execute('''INSERT INTO pending (name, description, category, location, uploadedBy, license, date, status)
                     VALUES (?, ?, ?, ?, ?, ?, DATE('now'), 'pending')''',
                  (artifact.get('Name', 'Unnamed Artifact'), artifact.get('Description', ''), artifact.get('Category', ''), artifact.get('Location', ''), uploadedBy, license))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/artifacts/mine', methods=['GET'])
def get_my_submissions():
    user = request.args.get('user')
    conn = get_db()
    c = conn.cursor()
    # Get from pending
    c.execute('SELECT * FROM pending WHERE uploadedBy=?', (user,))
    pending_rows = [dict(row) for row in c.fetchall()]
    # Get from approved
    c.execute('SELECT * FROM approved WHERE uploadedBy=?', (user,))
    approved_rows = [dict(row) for row in c.fetchall()]
    conn.close()
    # Mark status for frontend
    for row in pending_rows:
        row['status'] = row.get('status', 'pending')
    for row in approved_rows:
        row['status'] = 'approved'
    return jsonify(pending_rows + approved_rows)

@app.route('/api/artifacts/pending/<int:artifact_id>', methods=['GET'])
def get_pending_one(artifact_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM pending WHERE id=?', (artifact_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(dict(row))


@app.route('/api/artifacts/<int:artifact_id>', methods=['PUT'])
def update_rejected_artifact(artifact_id):
    """Allow the submitter to edit a rejected artifact and resubmit it.

    Supports:
      - application/json
      - multipart/form-data (for optional image re-upload)

    Rules:
      - Only artifacts in the `pending` table can be edited.
      - Only status='rejected' can be updated through this endpoint.
      - After update, status is reset to 'pending' and rejectionReason cleared.
    """

    is_multipart = bool(request.content_type and request.content_type.startswith('multipart/form-data'))
    if is_multipart:
        data = request.form
    else:
        data = request.json or {}

    conn = get_db()
    c = conn.cursor()

    c.execute('SELECT * FROM pending WHERE id=?', (artifact_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'Not found'}), 404

    if row['status'] != 'rejected':
        conn.close()
        return jsonify({'error': 'Only rejected artifacts can be edited.'}), 400

    # Optional: basic ownership check
    requested_user = ((data.get('uploadedBy') if hasattr(data, 'get') else None) or '').strip()
    if requested_user and row['uploadedBy'] and requested_user != row['uploadedBy']:
        conn.close()
        return jsonify({'error': 'You can only edit your own submissions.'}), 403

    name = (data.get('name') if hasattr(data, 'get') else None) or row['name']
    description = (data.get('description') if hasattr(data, 'get') else None) or row['description']
    category = (data.get('category') if hasattr(data, 'get') else None) or row['category']
    location = (data.get('location') if hasattr(data, 'get') else None) or row['location']

    # Optional: accept new images, save into backend/data
    # (we still don't store image metadata in DB in this prototype)
    image_filenames = []
    if is_multipart and 'images' in request.files:
        images = request.files.getlist('images')
        upload_folder = os.path.join(os.path.dirname(__file__), 'static', 'images')
        os.makedirs(upload_folder, exist_ok=True)
        for img in images:
            if img and img.filename:
                filename = secure_filename(img.filename)
                save_path = os.path.join(upload_folder, filename)
                img.save(save_path)
                image_filenames.append(filename)

    c.execute('''UPDATE pending
                 SET name=?, description=?, category=?, location=?, status='pending', rejectionReason=NULL
                 WHERE id=?''',
              (name, description, category, location, artifact_id))

    conn.commit()
    conn.close()
    return jsonify({'success': True, 'id': artifact_id, 'images': image_filenames})

@app.route('/api/search/suggest', methods=['GET'])
def suggest_artifacts():
    """Prefix-based suggestions for the search bar.

    Returns a small list of artifact names that start with the given prefix.
    This is intentionally simple (SQLite LIKE query) and fast.
    """
    prefix = (request.args.get('q') or '').strip()
    limit = int(request.args.get('limit') or 10)

    if not prefix:
        return jsonify({'suggestions': []})

    # Prevent pathological limits
    limit = max(1, min(limit, 25))

    conn = get_db()
    c = conn.cursor()

    # Case-insensitive prefix match. Escape % and _ so user input can't act as wildcards.
    escaped = prefix.replace('%', r'\%').replace('_', r'\_')
    like = f"{escaped}%"

    c.execute(
        """
        SELECT id, name
        FROM approved
        WHERE name LIKE ? ESCAPE '\\'
        ORDER BY name COLLATE NOCASE
        LIMIT ?
        """,
        (like, limit),
    )

    suggestions = [dict(row) for row in c.fetchall()]
    conn.close()

    return jsonify({'suggestions': suggestions})

# --- UI Route ---
@app.route('/')
def index():
    # Serve the main single-page UI from backend/templates/archSearch.html
    return render_template('archSearch.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)
