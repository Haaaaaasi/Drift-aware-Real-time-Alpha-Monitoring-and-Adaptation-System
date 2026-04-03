from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class DolphinDBSettings(BaseSettings):
    host: str = "localhost"
    port: int = 8848
    user: str = "admin"
    password: str = "123456"

    model_config = {"env_prefix": "DOLPHINDB_"}


class PostgresSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    db: str = "darams"
    user: str = "darams"
    password: str = "darams_dev"

    model_config = {"env_prefix": "POSTGRES_"}

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def async_dsn(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class RedisSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6379

    model_config = {"env_prefix": "REDIS_"}

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}"


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    log_level: str = "INFO"
    bar_type: str = "daily"
    universe_size: int = 50

    dolphindb: DolphinDBSettings = DolphinDBSettings()
    postgres: PostgresSettings = PostgresSettings()
    redis: RedisSettings = RedisSettings()

    mlflow_tracking_uri: str = "http://localhost:5000"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
