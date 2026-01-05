import os
import sqlite3
import numpy as np
import faiss
from PIL import Image
import torch
import torchvision.models as models

DB_PATH = os.path.join(os.path.dirname(__file__), 'archsearch.db')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'artifact_images.index')


def get_model():
    # Reduce thread-related instability on some macOS setups
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    w = models.ResNet18_Weights.DEFAULT
    m = models.resnet18(weights=w)
    m.fc = torch.nn.Identity()
    m.eval()
    transform = w.transforms()
    return m, transform


def image_files(images_field: str):
    if not images_field:
        return []
    return [p.strip() for p in images_field.split(',') if p.strip()]


def embed_image(model, transform, img: Image.Image) -> np.ndarray:
    img = img.convert('RGB')
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        v = model(t)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
    return v.cpu().numpy().astype('float32')[0]


def main():
    model, transform = get_model()
    dim = 512

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT id, images FROM approved')
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    static_dir = os.path.join(os.path.dirname(__file__), 'static', 'images')

    embeddings = []
    ids = []

    for r in rows:
        art_id = int(r['id'])
        files = image_files(r.get('images'))
        if not files:
            continue

        vecs = []
        for fn in files:
            path = os.path.join(static_dir, fn)
            if not os.path.exists(path):
                continue
            try:
                img = Image.open(path)
                vecs.append(embed_image(model, transform, img))
            except Exception as e:
                print(f"Skipping image for artifact {art_id} ({fn}): {e}")
                continue

        if not vecs:
            continue

        emb = np.mean(np.vstack(vecs), axis=0)
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        embeddings.append(emb.astype('float32'))
        ids.append(art_id)

    if embeddings:
        mat = np.vstack(embeddings).astype('float32')
        ids_np = np.array(ids, dtype=np.int64)
        index.add_with_ids(mat, ids_np)

    faiss.write_index(index, INDEX_PATH)
    print(f'Rebuilt image index: {os.path.basename(INDEX_PATH)}')
    print(f'  vectors: {index.ntotal}')
    print(f'  dim: {dim}')


if __name__ == '__main__':
    main()
