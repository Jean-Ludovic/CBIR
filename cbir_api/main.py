from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image
import io

from models import ImageResult

app = FastAPI(
    title="CBIR Microservice",
    description="API de recherche d'images par contenu (CBIR)",
    version="0.1.0",
)

# CORS (à adapter si tu as un vrai frontend plus tard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tu pourras restreindre à ton domaine ensuite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Simple endpoint pour vérifier que l'API tourne.
    """
    return {"status": "ok", "service": "cbir_api"}


@app.post("/search/text", response_model=List[ImageResult])
async def search_by_text(query: str, limit: int = 5):
    """
    Recherche fake par texte.
    Plus tard: transformer 'query' en embedding, chercher dans FAISS/pgvector.
    """
    # Pour l'instant, on renvoie des données factices
    results = [
        ImageResult(
            id=i,
            score=1.0 - i * 0.1,
            image_url=f"https://example.com/images/{i}.jpg",
            thumbnail_url=f"https://example.com/thumbnails/{i}.jpg",
            metadata={"query": query, "dummy": "true"},
        )
        for i in range(limit)
    ]
    return results


@app.post("/search/image", response_model=List[ImageResult])
async def search_by_image(file: UploadFile = File(...), limit: int = 5):
    """
    Recherche fake par image.
    Plus tard: extraire embedding de l'image, interroger index vectoriel.
    """
    # Vérif simple du type MIME
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    # Lecture de l'image (on ne fait rien avec pour l'instant, juste un check)
    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
        width, height = image.size
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image envoyée.")

    # Résultat factice
    results = [
        ImageResult(
            id=i,
            score=1.0 - i * 0.1,
            image_url=f"https://example.com/images/{i}.jpg",
            thumbnail_url=f"https://example.com/thumbnails/{i}.jpg",
            metadata={
                "info": "demo result",
                "input_width": str(width),
                "input_height": str(height),
            },
        )
        for i in range(limit)
    ]
    return results


# Permet de lancer directement: python main.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,   # Django est déjà sur 8000
        reload=True,
    )
