"""Request-scoped read-only SQLite sessions and dependency-injected publication."""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy import inspect
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from visionops.api.publication import Publication, publication
from visionops.config import Settings, get_settings
from visionops.db.session import build_engine, build_session_factory

SettingsDependency = Annotated[Settings, Depends(get_settings)]


def get_publication(settings: SettingsDependency) -> Publication:
    return publication(settings.dashboard_manifest)


def get_database(settings: SettingsDependency) -> Iterator[Session | None]:
    engine = None
    session = None
    try:
        url = make_url(settings.database_url)
        if url.get_backend_name() == "sqlite" and url.database and url.database != ":memory:":
            path = Path(url.database).resolve()
            if path.is_file():
                # URI mode=ro prevents GETs from creating a missing database or writing to it.
                readonly = url.set(database=f"file:{path}", query={"mode": "ro", "uri": "true"})
                engine = build_engine(readonly.render_as_string(hide_password=False))
                required = {
                    "videos",
                    "experiments",
                    "encoding_configurations",
                    "encoding_results",
                    "quality_metrics",
                }
                if required <= set(inspect(engine).get_table_names()):
                    session = build_session_factory(engine)()
    except (ValueError, OSError, SQLAlchemyError):
        session = None
    try:
        yield session
    finally:
        if session is not None:
            session.close()
        if engine is not None:
            engine.dispose()


DatabaseDependency = Annotated[Session | None, Depends(get_database)]
PublicationDependency = Annotated[Publication, Depends(get_publication)]
