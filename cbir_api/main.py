# cbir_api/main.py
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from .database import get_db, init_db
from .models import GalleryImage
from .features import extract_embedding
from .faiss_index import add_embedding, search_similar


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


@app.on_event("startup")
def on_startup():
    # création des tables si besoin
    init_db()


@app.get("/")
def root():
    return {"message": "CBIR API OK"}


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

    # 3) Créer la ligne en DB (sans embedding)
    img_db = GalleryImage(
        user_id=user_id,
        name=name,
        description=description,
        image_path=str(filepath)  # ex: "uploads/gallery/user_1_1.jpg"
    )
    db.add(img_db)
    db.commit()
    db.refresh(img_db)

    # 4) Extraire embedding et l’ajouter dans FAISS
    embedding = extract_embedding(str(filepath))
    add_embedding(img_db.id, embedding)

    # 5) Retourner les infos pour le frontend
    return {
        "id": img_db.id,
        "name": img_db.name,
        "description": img_db.description,
        # URL pour afficher dans Django : http://127.0.0.1:8001/media/gallery/xxx.jpg
        "image_url": f"/media/gallery/{filename}",
    }


@app.post("/gallery/search")
async def search_gallery(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Utilise la nouvelle image comme requête,
    cherche les images les plus similaires dans la galerie de CE user.
    """
    # 1) Sauvegarder l'image de requête
    ext = file.filename.split(".")[-1]
    query_filename = f"user_{user_id}_query.{ext}"
    query_path = QUERY_DIR / query_filename
    with open(query_path, "wb") as f:
        f.write(await file.read())

    # 2) Extraire l'embedding de la requête
    query_embedding = extract_embedding(str(query_path))

    # 3) Recherche FAISS globale (on prend plus large, on filtrera après)
    distances, ids = search_similar(query_embedding, k=20)

    # 4) Filtrer sur les images appartenant à cet utilisateur
    results = []
    for dist, img_id in zip(distances, ids):
        if img_id == -1:
            continue

        img = db.query(GalleryImage).filter_by(id=int(img_id), user_id=user_id).first()
        if not img:
            continue

        # On reconstruit l'URL publique pour l'image
        filename = Path(img.image_path).name  # "user_1_1.jpg"
        image_url = f"/media/gallery/{filename}"

        results.append({
            "image_id": img.id,
            "name": img.name,
            "description": img.description,
            "image_path": image_url,
            "distance": float(dist),
        })

    # 5) Trier par distance croissante (plus proche d'abord)
    results.sort(key=lambda x: x["distance"])

    return {
        "query_image": f"/media/query/{query_filename}",
        "results": results,
    }
