# cbir_api/main.py
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .database import get_db, init_db
from .models import GalleryImage
from .features import extract_embedding
from .faiss_index import add_embedding  # on garde FAISS pour l'index


app = FastAPI()

# Dossiers uploads
BASE_UPLOAD_DIR = Path("uploads")
GALLERY_DIR = BASE_UPLOAD_DIR / "gallery"
QUERY_DIR = BASE_UPLOAD_DIR / "query"

GALLERY_DIR.mkdir(parents=True, exist_ok=True)
QUERY_DIR.mkdir(parents=True, exist_ok=True)

# Expose /media pour servir les images
app.mount("/media", StaticFiles(directory=str(BASE_UPLOAD_DIR)), name="media")

# CORS : Django sur 8000, FastAPI sur 8001
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_current_user_id():
    # TODO: brancher avec ta vraie auth Django plus tard
    return 1


def compute_distances(vec_a: np.ndarray, vec_b: np.ndarray):
    """
    Calcule plusieurs distances entre deux vecteurs :
    euclidienne, Manhattan, Chebyshev, Canberra.
    """
    diff = vec_a - vec_b
    abs_diff = np.abs(diff)

    d_euclid = float(np.linalg.norm(diff))
    d_manhattan = float(np.sum(abs_diff))
    d_chebyshev = float(np.max(abs_diff))
    d_canberra = float(
        np.sum(
            abs_diff / (np.abs(vec_a) + np.abs(vec_b) + 1e-8)
        )
    )

    return {
        "euclidean": d_euclid,
        "manhattan": d_manhattan,
        "chebyshev": d_chebyshev,
        "canberra": d_canberra,
    }


@app.on_event("startup")
def on_startup():
    # création des tables si besoin
    init_db()


@app.get("/")
def root():
    return {"message": "CBIR API OK"}


# ---------------------------------------------------------------------------
# 1) Upload dans la galerie
# ---------------------------------------------------------------------------
@app.post("/gallery/upload")
async def upload_gallery_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    # 1) Vérifier limite 5 images
    count = db.query(GalleryImage).filter_by(user_id=user_id).count()
    if count >= 5:
        raise HTTPException(
            status_code=400,
            detail="Tu as déjà 5 images dans ta galerie."
        )

    # 2) Sauvegarder l'image
    ext = file.filename.split(".")[-1]
    filename = f"user_{user_id}_{count + 1}.{ext}"
    filepath = GALLERY_DIR / filename
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # 3) Créer la ligne en DB
    img_db = GalleryImage(
        user_id=user_id,
        name=name,
        description=description,
        image_path=str(filepath),  # ex: "uploads/gallery/user_1_1.jpg"
    )
    db.add(img_db)
    db.commit()
    db.refresh(img_db)

    # 4) Extraire embedding et l’ajouter dans FAISS (pour la roadmap)
    embedding = extract_embedding(str(filepath))
    add_embedding(img_db.id, embedding)

    # 5) Retourner les infos pour le frontend
    return {
        "id": img_db.id,
        "name": img_db.name,
        "description": img_db.description,
        # URL publique : http://127.0.0.1:8001/media/gallery/xxx.jpg
        "image_url": f"/media/gallery/{filename}",
    }


# ---------------------------------------------------------------------------
# 2) Liste de la galerie (vue CRUD + stats)
# ---------------------------------------------------------------------------
from pathlib import Path as SysPath  # pour manipuler les noms de fichier

@app.get("/gallery")
def list_gallery(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    images = (
        db.query(GalleryImage)
        .filter(GalleryImage.user_id == user_id)
        .order_by(GalleryImage.id.asc())
        .all()
    )

    results = []
    for img in images:
        filename = SysPath(img.image_path).name  # ex: user_1_1.jpg
        image_url = f"/media/gallery/{filename}"

        results.append({
            "id": img.id,
            "name": img.name,
            "description": img.description,
            "image_url": image_url,
        })

    return results


# ---------------------------------------------------------------------------
# 3) Recherche dans la galerie = distances + ranking Top 1 / Top 2 / ...
# ---------------------------------------------------------------------------
@app.post("/gallery/search")
async def search_gallery(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Utilise la nouvelle image comme requête,
    calcule plusieurs distances par rapport à TOUTES les images de la galerie
    de cet utilisateur.
    """
    # 1) Sauvegarder l'image de requête
    ext = file.filename.split(".")[-1]
    query_filename = f"user_{user_id}_query.{ext}"
    query_path = QUERY_DIR / query_filename
    with open(query_path, "wb") as f:
        f.write(await file.read())

    # 2) Embedding de la requête (forcé en np.array float32)
    query_emb_raw = extract_embedding(str(query_path))
    query_emb = np.array(query_emb_raw, dtype="float32")

    # 3) Récupérer toutes les images de la galerie de cet user
    images = (
        db.query(GalleryImage)
        .filter(GalleryImage.user_id == user_id)
        .order_by(GalleryImage.id.asc())
        .all()
    )

    if not images:
        return {
            "query_image": f"/media/query/{query_filename}",
            "results": [],
        }

    results = []
    for img in images:
        img_emb_raw = extract_embedding(img.image_path)
        img_emb = np.array(img_emb_raw, dtype="float32")

        distances = compute_distances(query_emb, img_emb)

        filename = Path(img.image_path).name
        image_url = f"/media/gallery/{filename}"

        results.append({
            "image_id": img.id,
            "name": img.name,
            "description": img.description,
            "image_path": image_url,
            "distances": distances,
        })

    # Tri par défaut : distance euclidienne
    results.sort(key=lambda x: x["distances"]["euclidean"])

    return {
        "query_image": f"/media/query/{query_filename}",
        "results": results,
    }


# ---------------------------------------------------------------------------
# 4) Metrics entre toutes les paires d'images (vue "mode expert")
# ---------------------------------------------------------------------------
@app.get("/gallery/metrics")
def gallery_metrics(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Calcule les distances entre toutes les paires d'images de la galerie
    de l'utilisateur.
    """
    images = (
        db.query(GalleryImage)
        .filter(GalleryImage.user_id == user_id)
        .order_by(GalleryImage.id.asc())
        .all()
    )

    if len(images) < 2:
        return []  # frontend affichera un message

    # Pré-calcul des embeddings pour chaque image
    embs = []
    for img in images:
        emb_raw = extract_embedding(img.image_path)
        emb = np.array(emb_raw, dtype="float32")
        embs.append(emb)

    rows = []
    n = len(images)
    for i in range(n):
        for j in range(i + 1, n):
            dists = compute_distances(embs[i], embs[j])
            rows.append({
                "a_name": images[i].name,
                "b_name": images[j].name,
                "d_euclid": dists["euclidean"],
                "d_manhattan": dists["manhattan"],
                "d_chebyshev": dists["chebyshev"],
                "d_canberra": dists["canberra"],
            })

    return rows
