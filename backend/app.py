import os
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np

# --- Config ---
DB_PATH = os.path.join(os.path.dirname(__file__), 'archsearch.db')

# --- Flask Setup ---
app = Flask(__name__)
# For a local prototype (including running the frontend via file:// which uses Origin: null),
# allow all origins on API routes.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- SQLite Setup ---
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
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
        citation TEXT
    )''')
    conn.commit()
    conn.close()

# --- FAISS Setup ---
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss.index')
EMBEDDING_DIM = 384  # e.g., for MiniLM

def get_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        return faiss.read_index(FAISS_INDEX_PATH)
    else:
        return faiss.IndexFlatL2(EMBEDDING_DIM)

def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_PATH)

# --- Dummy Embedding (replace with real model) ---
# NOTE: The previous implementation used numpy random seeded by Python's hash().
# Python's hash() is salted per-process, so embeddings changed on every restart,
# causing "alternating"/unstable results.

def _stable_hash_to_uint32(text: str) -> int:
    """Stable 32-bit hash for deterministic embeddings across restarts."""
    import hashlib

    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little", signed=False)


def embed_text(text):
    """Deterministic placeholder embedding.

    This uses a simple character 3-gram hashing scheme so that similar strings
    (e.g., same name/keywords) produce similar vectors. Replace with
    sentence-transformers in production.
    """
    raw = (text or "").strip().lower()
    if not raw:
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)

    v = np.zeros((EMBEDDING_DIM,), dtype=np.float32)
    padded = f"  {raw}  "
    for i in range(len(padded) - 2):
        tri = padded[i : i + 3]
        h = _stable_hash_to_uint32(tri)
        idx = h % EMBEDDING_DIM
        v[idx] += 1.0

    # L2 normalize
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v /= norm
    return v

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
        for img in images:
            filename = img.filename
            save_path = os.path.join(os.path.dirname(__file__), '../data', filename)
            img.save(save_path)
            image_filenames.append(filename)
    conn = get_db()
    c = conn.cursor()
    c.execute('''INSERT INTO pending (name, description, category, location, uploadedBy, license, date, status, rejectionReason)
                 VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', NULL)''',
              (name, description, category, location, uploadedBy, license, date))
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
    # Move to approved
    c.execute('''INSERT INTO approved (name, description, category, location, uploadedBy, license, date, citation)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (row['name'], row['description'], row['category'], row['location'], row['uploadedBy'], row['license'], row['date'], 'Verified by Expert Panel'))
    c.execute('DELETE FROM pending WHERE id=?', (artifact_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

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

    if not query:
        return jsonify({'results': []})

    # Load approved artifacts
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM approved')
    artifacts = [dict(row) for row in c.fetchall()]
    conn.close()

    if not artifacts:
        return jsonify({'results': []})

    # Always rebuild the index from the current DB (small prototype; keeps it correct)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)

    def _artifact_text(a: dict) -> str:
        # Use multiple fields so searches work even with short/odd descriptions.
        return ' '.join([
            str(a.get('name') or ''),
            str(a.get('description') or ''),
            str(a.get('category') or ''),
            str(a.get('location') or ''),
        ]).strip()

    vectors = np.stack([embed_text(_artifact_text(a)) for a in artifacts]).astype('float32')
    index.add(vectors)
    save_faiss_index(index)

    # Search
    q_emb = embed_text(query)
    k = min(5, len(artifacts))
    D, I = index.search(np.expand_dims(q_emb, 0), k)

    # Convert L2 distance to a rough similarity score in [0,1].
    # (score = 1/(1+d))
    best_i = int(I[0][0])
    best_d = float(D[0][0])
    best_score = 1.0 / (1.0 + best_d)

    # Heuristic threshold to avoid returning "some" result for irrelevant queries.
    # Tune as needed.
    MIN_SCORE = 0.48

    if best_score < MIN_SCORE or best_i >= len(artifacts):
        return jsonify({'results': []})

    # Return only the best match (frontend expects/uses top result)
    return jsonify({'results': [artifacts[best_i]]})

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

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host="127.0.0.1", port=5000)
