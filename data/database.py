"""
Trading Bot - Database Manager
Handles SQLite connection, session management, and initialization.
"""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from config.settings import BASE_DIR
from data.models import Base


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(BASE_DIR / "data" / "trading_bot.db")

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(self.engine)
        logger.info(f"Database initialized at {db_path}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def close(self):
        """Close engine connections."""
        self.engine.dispose()
        logger.info("Database connections closed")


# Global database instance
db = Database()
