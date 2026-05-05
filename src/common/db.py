"""Database connection management for PostgreSQL, DolphinDB, and Redis."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import dolphindb as ddb
import psycopg2
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import get_settings

_settings = get_settings()

# --- PostgreSQL (SQLAlchemy) ---

_pg_engine = create_engine(_settings.postgres.dsn, pool_size=5, max_overflow=10)
_SessionLocal = sessionmaker(bind=_pg_engine, autocommit=False, autoflush=False)


@contextmanager
def get_pg_session() -> Generator[Session, None, None]:
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_pg_connection():
    return psycopg2.connect(
        host=_settings.postgres.host,
        port=_settings.postgres.port,
        dbname=_settings.postgres.db,
        user=_settings.postgres.user,
        password=_settings.postgres.password,
    )


# --- DolphinDB ---

_ddb_session: ddb.Session | None = None


def get_dolphindb() -> ddb.Session:
    global _ddb_session
    if _ddb_session is None or _ddb_session.isClosed():
        _ddb_session = ddb.Session()
        _ddb_session.connect(
            _settings.dolphindb.host,
            _settings.dolphindb.port,
            _settings.dolphindb.user,
            _settings.dolphindb.password,
        )
    return _ddb_session


# --- Redis ---

_redis_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=_settings.redis.host,
            port=_settings.redis.port,
            decode_responses=True,
        )
    return _redis_client
