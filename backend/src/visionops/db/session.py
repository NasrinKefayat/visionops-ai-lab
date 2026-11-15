"""SQLAlchemy engine and session factory without import-time database creation."""

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from visionops.config import get_settings


class Base(DeclarativeBase):
    pass


def build_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_settings().database_url
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite:") else {}
    engine = create_engine(url, **kwargs)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _record) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
