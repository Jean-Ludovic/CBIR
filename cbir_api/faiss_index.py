# faiss_index.py
import faiss
import numpy as np
from pathlib import Path

INDEX_PATH = Path("data/faiss_gallery.index")
EMBEDDING_DIM = 512  # à adapter selon ton modèle

_index = None

def get_index():
    global _index
    if _index is None:
        if INDEX_PATH.exists():
            _index = faiss.read_index(str(INDEX_PATH))
        else:
            base_index = faiss.IndexFlatL2(EMBEDDING_DIM)
            _index = faiss.IndexIDMap(base_index)
    return _index

def save_index():
    global _index
    if _index is not None:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(_index, str(INDEX_PATH))

def add_embedding(image_id: int, embedding: np.ndarray):
    index = get_index()
    # embedding: shape (EMBEDDING_DIM,) -> (1, d)
    emb = embedding.astype("float32").reshape(1, -1)
    ids = np.array([image_id], dtype="int64")
    index.add_with_ids(emb, ids)
    save_index()

def search_similar(embedding: np.ndarray, k: int = 5):
    index = get_index()
    if index.ntotal == 0:
        return [], []
    emb = embedding.astype("float32").reshape(1, -1)
    distances, ids = index.search(emb, k)
    return distances[0], ids[0]
