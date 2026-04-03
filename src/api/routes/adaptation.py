"""API routes for adaptation control."""

from __future__ import annotations

from fastapi import APIRouter

from src.adaptation.model_registry import ModelRegistryManager

router = APIRouter()


@router.get("/models")
def list_models(status: str | None = None):
    """List registered models."""
    mgr = ModelRegistryManager()
    df = mgr.get_all_models(status=status)
    return df.to_dict(orient="records")


@router.get("/models/production")
def get_production_model():
    """Get current production model."""
    mgr = ModelRegistryManager()
    model = mgr.get_production_model()
    return model or {"message": "No production model"}


@router.post("/models/{model_id}/promote")
def promote_model(model_id: str):
    """Promote a shadow model to production."""
    mgr = ModelRegistryManager()
    mgr.promote_model(model_id)
    return {"message": f"Model {model_id} promoted to production"}


@router.get("/regime-pool")
def get_regime_pool():
    """List all regimes in the concept pool."""
    import pandas as pd
    from src.common.db import get_pg_connection

    conn = get_pg_connection()
    try:
        df = pd.read_sql("SELECT * FROM regime_pool ORDER BY detected_at DESC", conn)
        return df.to_dict(orient="records")
    finally:
        conn.close()
