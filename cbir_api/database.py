# cbir_api/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base  # on va le définir juste après

# Pour commencer simple : SQLite local
SQLALCHEMY_DATABASE_URL = "sqlite:///./cbir_gallery.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
