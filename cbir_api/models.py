from pydantic import BaseModel
from typing import Optional, Dict


class ImageResult(BaseModel):
    id: int
    score: float
    image_url: str
    thumbnail_url: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
