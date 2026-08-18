"""Tests integración — lectura de clusters contra postgres real.

Regresión del bug read_cluster(include_members) → HTTP 500 (2026-08-18):
    GET /api/v1/clusters/{id}/members hacía
        md = (cluster.get("metadata") or {}).get("member_distances", {})
    y cluster["metadata"] vuelve de asyncpg como STRING JSON (JSONB sin codec),
    así que .get() lanzaba AttributeError: 'str' object has no attribute 'get'
    en todo cluster con metadata no-nula (todos los cell-generated).
    Fix: usar el helper _parse_jsonb (clusters.py). Este test salta a rojo si
    alguien revierte esa línea.

Run desde api/:  python -m pytest tests/test_clusters.py -v
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
import asyncpg
from fastapi.testclient import TestClient

from conftest import TEST_DB_URL
os.environ.setdefault("DATABASE_URL", TEST_DB_URL)
os.environ.setdefault("ENVIRONMENT", "development")

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import create_app
from auth import generate_api_key


@pytest.fixture(scope="module")
def app():
    return create_app("development")


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def super_jwt(client):
    """Crea un api_key de superusuario, lo cambia por JWT, y limpia al salir."""
    key_plain, key_hash = generate_api_key()

    async def _setup():
        conn = await asyncpg.connect(dsn=os.environ["DATABASE_URL"])
        try:
            row = await conn.fetchrow(
                "INSERT INTO api_keys (key_hash, name, user_id, active) "
                "VALUES ($1, 'pytest-clusters', 1, true) RETURNING id",
                key_hash,
            )
            return row["id"]
        finally:
            await conn.close()

    async def _cleanup(key_id):
        conn = await asyncpg.connect(dsn=os.environ["DATABASE_URL"])
        try:
            await conn.execute("DELETE FROM api_keys WHERE id = $1", key_id)
        finally:
            await conn.close()

    key_id = asyncio.run(_setup())
    token = client.post("/auth/token", json={"api_key": key_plain}).json()["access_token"]
    yield token
    asyncio.run(_cleanup(key_id))


@pytest.fixture(scope="module")
def cluster_with_metadata():
    """Un cluster REAL con metadata no-nula y miembros — la condición que
    disparaba el 500. Salta el test si el grafo aún no tiene clusters así."""
    async def _find():
        conn = await asyncpg.connect(dsn=os.environ["DATABASE_URL"])
        try:
            return await conn.fetchval(
                "SELECT id FROM memory_clusters "
                "WHERE metadata IS NOT NULL AND metadata::text <> '{}' "
                "AND coalesce(array_length(member_ids, 1), 0) > 0 "
                "AND status = 'active' LIMIT 1"
            )
        finally:
            await conn.close()

    cid = asyncio.run(_find())
    if cid is None:
        pytest.skip("no hay clusters con metadata + miembros en la DB")
    return str(cid)


def auth(jwt):
    return {"Authorization": f"Bearer {jwt}"}


# ---------------------------------------------------------------------------
# GET /api/v1/clusters/{id}/members
# ---------------------------------------------------------------------------

def test_members_of_cluster_with_metadata_returns_200(client, super_jwt, cluster_with_metadata):
    """Regresión del 500: metadata JSONB devuelta como str no debe romper el
    endpoint. Antes del fix esto daba 500 (AttributeError en member_distances)."""
    r = client.get(
        f"/api/v1/clusters/{cluster_with_metadata}/members",
        headers=auth(super_jwt),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "members" in body
    assert "total" in body
    assert isinstance(body["members"], list)


def test_members_of_nonexistent_cluster_returns_404(client, super_jwt):
    """Cluster inexistente → 404 limpio, nunca 500."""
    r = client.get(
        "/api/v1/clusters/00000000-0000-0000-0000-000000000000/members",
        headers=auth(super_jwt),
    )
    assert r.status_code == 404, r.text
