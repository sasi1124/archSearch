import os
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = os.path.join(os.path.dirname(__file__), 'archsearch.db')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'artifact_vectors.index')
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def main():
    model = SentenceTransformer(MODEL_NAME)
    dim = int(model.get_sentence_embedding_dimension())

    # Create cosine-similarity index via inner product on normalized embeddings
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, name, description FROM approved')
    rows = c.fetchall()
    conn.close()

    if not rows:
        # Save empty index
        faiss.write_index(index, INDEX_PATH)
        print('No approved artifacts found. Wrote empty index.')
        return

    ids = np.array([int(r['id']) for r in rows], dtype=np.int64)
    texts = [f"{r['name']} {r['description'] or ''}".strip() for r in rows]

    # Compute embeddings in batch
    vecs = model.encode(texts, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype=np.float32)

    index.add_with_ids(vecs, ids)
    faiss.write_index(index, INDEX_PATH)

    print(f'Rebuilt index: {os.path.basename(INDEX_PATH)}')
    print(f'  vectors: {index.ntotal}')
    print(f'  dim: {dim}')


if __name__ == '__main__':
    main()
