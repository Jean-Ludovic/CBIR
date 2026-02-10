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
from .faiss_index import add_embedding


app = FastAPI(title="CBIR API")

# ----------------------------
# Upload folders
# ----------------------------
BASE_UPLOAD_DIR = Path("uploads")
GALLERY_DIR = BASE_UPLOAD_DIR / "gallery"
QUERY_DIR = BASE_UPLOAD_DIR / "query"

GALLERY_DIR.mkdir(parents=True, exist_ok=True)
QUERY_DIR.mkdir(parents=True, exist_ok=True)

# Serve images at /media/...
app.mount("/media", StaticFiles(directory=str(BASE_UPLOAD_DIR)), name="media")

# ----------------------------
# CORS (optionnel ici, mais ok)
# ----------------------------
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


def get_current_user_id():
    # TODO: brancher vraie auth plus tard
    return 1


def compute_distances(vec_a: np.ndarray, vec_b: np.ndarray):
    diff = vec_a - vec_b
    abs_diff = np.abs(diff)

    d_euclid = float(np.linalg.norm(diff))
    d_manhattan = float(np.sum(abs_diff))
    d_chebyshev = float(np.max(abs_diff))
    d_canberra = float(np.sum(abs_diff / (np.abs(vec_a) + np.abs(vec_b) + 1e-8)))

    return {
        "euclidean": d_euclid,
        "manhattan": d_manhattan,
        "chebyshev": d_chebyshev,
        "canberra": d_canberra,
    }


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


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
    count = db.query(GalleryImage).filter_by(user_id=user_id).count()
    if count >= 10:
        raise HTTPException(status_code=400, detail="Tu as déjà 10 images dans ta galerie.")

    ext = file.filename.split(".")[-1]
    filename = f"user_{user_id}_{count + 1}.{ext}"
    filepath = GALLERY_DIR / filename

    with open(filepath, "wb") as f:
        f.write(await file.read())

    img_db = GalleryImage(
        user_id=user_id,
        name=name,
        description=description,
        image_path=str(filepath),
    )
    db.add(img_db)
    db.commit()
    db.refresh(img_db)

    embedding = extract_embedding(str(filepath))
    add_embedding(img_db.id, embedding)

    return {
        "id": img_db.id,
        "name": img_db.name,
        "description": img_db.description,
        "image_url": f"/media/gallery/{filename}",
    }


# ---------------------------------------------------------------------------
# 2) Liste de la galerie
# ---------------------------------------------------------------------------
from pathlib import Path as SysPath

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
        filename = SysPath(img.image_path).name
        results.append({
            "id": img.id,
            "name": img.name,
            "description": img.description,
            "image_url": f"/media/gallery/{filename}",
        })

    return results


# ---------------------------------------------------------------------------
# 3) Recherche
# ---------------------------------------------------------------------------
@app.post("/gallery/search")
async def search_gallery(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    ext = file.filename.split(".")[-1]
    query_filename = f"user_{user_id}_query.{ext}"
    query_path = QUERY_DIR / query_filename

    with open(query_path, "wb") as f:
        f.write(await file.read())

    query_emb_raw = extract_embedding(str(query_path))
    query_emb = np.array(query_emb_raw, dtype="float32")

    images = (
        db.query(GalleryImage)
        .filter(GalleryImage.user_id == user_id)
        .order_by(GalleryImage.id.asc())
        .all()
    )

    if not images:
        return {"query_image": f"/media/query/{query_filename}", "results": []}

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

    results.sort(key=lambda x: x["distances"]["euclidean"])

    return {
        "query_image": f"/media/query/{query_filename}",
        "results": results,
    }


# ---------------------------------------------------------------------------
# 4) Metrics (mode expert)
# ---------------------------------------------------------------------------
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

    embs = []
    for img in images:
        emb_raw = extract_embedding(img.image_path)
        embs.append(np.array(emb_raw, dtype="float32"))

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
