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
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'artifact_vectors.index')
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

    # Move to approved table and get the new ID
    c.execute('''INSERT INTO approved (name, description, category, location, uploadedBy, license, date, citation)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (row['name'], row['description'], row['category'], row['location'], row['uploadedBy'], row['license'], row['date'], 'Verified by Expert Panel'))
    new_id = c.lastrowid

    # Add to FAISS index
    index = get_faiss_index()
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
    k = data.get('k', 5) # Number of results to return

    if not query:
        return jsonify({'results': []})

    # Load index
    index = get_faiss_index()
    if index.ntotal == 0:
        return jsonify({'results': []})

    # Embed query and search
    query_embedding = embed_text(query)
    distances, ids = index.search(np.array([query_embedding], dtype=np.float32), k)

    # Get artifact details from DB
    if len(ids[0]) == 0:
        return jsonify({'results': []})

    conn = get_db()
    c = conn.cursor()
    # The IDs from FAISS are in a nested list, e.g., [[1, 5, 3]]
    found_ids = [int(i) for i in ids[0] if i >= 0] # FAISS can return -1 for no result
    if not found_ids:
        conn.close()
        return jsonify({'results': []})

    # Create a placeholder string for the IN clause
    placeholders = ','.join('?' for _ in found_ids)
    c.execute(f'SELECT * FROM approved WHERE id IN ({placeholders})', found_ids)
    artifacts = [dict(row) for row in c.fetchall()]
    conn.close()

    # Sort results by the order returned by FAISS
    id_to_artifact = {art['id']: art for art in artifacts}
    sorted_artifacts = [id_to_artifact[i] for i in found_ids if i in id_to_artifact]

    return jsonify({'results': sorted_artifacts})

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
