from sqlalchemy import create_engine, Column, Integer, String, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DB_FILE

DATABASE_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Favorite(Base):
    __tablename__ = "favorites"
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(String, index=True, nullable=False, unique=True)  # one user → track_id unique

def init_db():
    Base.metadata.create_all(bind=engine)

def add_favorite(track_id: str):
    with SessionLocal() as db:
        exists = db.query(Favorite).filter_by(track_id=track_id).first()
        if exists:
            return exists.id
        fav = Favorite(track_id=str(track_id))
        db.add(fav)
        db.commit()
        db.refresh(fav)
        return fav.id

def list_favorites():
    with SessionLocal() as db:
        rows = db.query(Favorite).all()
        return [{"id": r.id, "track_id": r.track_id} for r in rows]

def favorites_exists(track_id: str):
    with SessionLocal() as db:
        return db.query(Favorite).filter_by(track_id=track_id).first() is not None

def remove_favorite(track_id: str):
    with SessionLocal() as db:
        row = db.query(Favorite).filter_by(track_id=track_id).first()
        if row:
            db.delete(row)
            db.commit()
