# cbir_api/main.py
from pathlib import Path
import tempfile
import numpy as np
import requests

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_db, init_db
from .models import GalleryImage
from .features import extract_embedding
from .faiss_index import add_embedding


app = FastAPI(title="CBIR API (Hybrid: Local + URL)")

# ------------------------------------------------------------------
# Local storage (optionnel si tu utilises surtout Cloudinary)
# ------------------------------------------------------------------
BASE_UPLOAD_DIR = Path("uploads")
GALLERY_DIR = BASE_UPLOAD_DIR / "gallery"
QUERY_DIR = BASE_UPLOAD_DIR / "query"

GALLERY_DIR.mkdir(parents=True, exist_ok=True)
QUERY_DIR.mkdir(parents=True, exist_ok=True)

# Expose /media pour servir les images LOCALES
app.mount("/media", StaticFiles(directory=str(BASE_UPLOAD_DIR)), name="media")

# ------------------------------------------------------------------
# CORS (optionnel ici car Streamlit appelle côté serveur, mais safe)
# ------------------------------------------------------------------
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "https://moteurrechercheparcomparaison.streamlit.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------
def get_current_user_id():
    # TODO: brancher vraie auth plus tard
    return 1


def is_http_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


def norm_path(p: str) -> str:
    """Normalise un chemin Windows/Linux en POSIX (évite les \ et les 404)."""
    return (p or "").replace("\\", "/")


def filename_from_any_path(p: str) -> str:
    """Extrait le nom de fichier même si p contient des backslashes Windows."""
    return Path(norm_path(p)).name


def to_display_url(image_path_or_url: str) -> str:
    """
    - Si c'est une URL (Cloudinary), on renvoie direct.
    - Si c'est un chemin local, on renvoie /media/gallery/<filename>
    """
    if is_http_url(image_path_or_url):
        return image_path_or_url

    filename = filename_from_any_path(image_path_or_url)
    return f"/media/gallery/{filename}"


def embedding_from_source(image_source: str) -> np.ndarray:
    """
    image_source:
      - chemin local (uploads/...)
      - URL (Cloudinary)
    Retourne np.array float32.
    """
    if is_http_url(image_source):
        r = requests.get(image_source, timeout=30)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
            tmp.write(r.content)
            tmp.flush()
            emb = extract_embedding(tmp.name)
    else:
        emb = extract_embedding(norm_path(image_source))

    return np.array(emb, dtype="float32")


def compute_distances(vec_a: np.ndarray, vec_b: np.ndarray):
    diff = vec_a - vec_b
    abs_diff = np.abs(diff)

    return {
        "euclidean": float(np.linalg.norm(diff)),
        "manhattan": float(np.sum(abs_diff)),
        "chebyshev": float(np.max(abs_diff)),
        "canberra": float(np.sum(abs_diff / (np.abs(vec_a) + np.abs(vec_b) + 1e-8))),
    }


# ------------------------------------------------------------------
# Startup
# ------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "CBIR API OK"}


# ------------------------------------------------------------------
# Payload: Add by URL
# ------------------------------------------------------------------
class AddByUrlIn(BaseModel):
    url: str
    name: str
    description: str = ""


# ------------------------------------------------------------------
# 1) Upload fichier (local)
# ------------------------------------------------------------------
@app.post("/gallery/upload")
async def upload_gallery_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    # Limite
    count = db.query(GalleryImage).filter_by(user_id=user_id).count()
    if count >= 20:
        raise HTTPException(status_code=400, detail="Limite de 20 images atteinte.")

    # Sauvegarde locale
    ext = file.filename.split(".")[-1]
    filename = f"user_{user_id}_{count + 1}.{ext}"
    filepath = GALLERY_DIR / filename

    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Stockage DB: chemin POSIX (cross-platform)
    filepath_posix = filepath.as_posix()

    img_db = GalleryImage(
        user_id=user_id,
        name=name,
        description=description,
        image_path=filepath_posix,
    )
    db.add(img_db)
    db.commit()
    db.refresh(img_db)

    # Embedding + index
    emb = embedding_from_source(filepath_posix)
    add_embedding(img_db.id, emb)

    return {
        "id": img_db.id,
        "name": img_db.name,
        "description": img_db.description,
        "image_url": to_display_url(img_db.image_path),  # /media/...
    }


# ------------------------------------------------------------------
# 1B) Ajouter par URL (Cloudinary)
# ------------------------------------------------------------------
@app.post("/gallery/add_by_url")
def add_gallery_image_by_url(
    payload: AddByUrlIn,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    # Limite
    count = db.query(GalleryImage).filter_by(user_id=user_id).count()
    if count >= 20:
        raise HTTPException(status_code=400, detail="Limite de 20 images atteinte.")

    if not is_http_url(payload.url):
        raise HTTPException(status_code=400, detail="URL invalide (http/https requis).")

    # Stocker l'URL en DB
    img_db = GalleryImage(
        user_id=user_id,
        name=payload.name,
        description=payload.description,
        image_path=payload.url,  # ✅ Cloudinary
    )
    db.add(img_db)
    db.commit()
    db.refresh(img_db)

    # Embedding + index
    emb = embedding_from_source(payload.url)
    add_embedding(img_db.id, emb)

    return {
        "id": img_db.id,
        "name": img_db.name,
        "description": img_db.description,
        "image_url": payload.url,  # ✅ direct
    }


# ------------------------------------------------------------------
# 2) Liste galerie (URL ou local)
# ------------------------------------------------------------------
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
        results.append({
            "id": img.id,
            "name": img.name,
            "description": img.description,
            "image_url": to_display_url(img.image_path),
            "source": "url" if is_http_url(img.image_path) else "local",
        })

    return results


# ------------------------------------------------------------------
# 3) Recherche CBIR (query upload local, comparaison local+url)
# ------------------------------------------------------------------
@app.post("/gallery/search")
async def search_gallery(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    # Sauvegarder query localement (temp “persistant” dans uploads/query)
    ext = file.filename.split(".")[-1]
    query_filename = f"user_{user_id}_query.{ext}"
    query_path = QUERY_DIR / query_filename

    with open(query_path, "wb") as f:
        f.write(await file.read())

    query_emb = embedding_from_source(query_path.as_posix())

    # Récupérer galerie
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
        img_emb = embedding_from_source(img.image_path)
        distances = compute_distances(query_emb, img_emb)

        results.append({
            "image_id": img.id,
            "name": img.name,
            "description": img.description,
            "image_path": to_display_url(img.image_path),
            "source": "url" if is_http_url(img.image_path) else "local",
            "distances": distances,
        })

    results.sort(key=lambda x: x["distances"]["euclidean"])

    return {
        "query_image": f"/media/query/{query_filename}",
        "results": results,
    }


# ------------------------------------------------------------------
# 4) Metrics (mode expert) (local+url)
# ------------------------------------------------------------------
@app.get("/gallery/metrics")
def gallery_metrics(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    images = (
        db.query(GalleryImage)
        .filter(GalleryImage.user_id == user_id)
        .order_by(GalleryImage.id.asc())
        .all()
    )

    if len(images) < 2:
        return []

    # Pré-calc embeddings
    embs = [embedding_from_source(img.image_path) for img in images]

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
