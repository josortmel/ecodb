"""Clusters endpoint — metacognicion v2.0 §7.3 + Memory Agent v1.3 search+telescopic."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from auth import get_current_user
from db import get_pool
from embeddings_client import embed_text
from events import broadcast_event
from pagination import paginate
from permissions import (
    precompute_read_visibility,
    check_read_memory,
    resolve_agent_for_actor,
    resolve_cluster_for_actor,
)

_log = logging.getLogger("ecodb.clusters")


router = APIRouter(prefix="/clusters", tags=["clusters"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ClusterSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    agent_id: int
    level: str
    label: str
    detail: Optional[str] = None
    narrative: Optional[str] = None
    member_count: int
    source_count: int
    pattern_flags: dict = Field(default_factory=dict)
    period_start: date
    period_end: date
    status: str
    narrated_at: Optional[datetime] = None
    created_at: datetime


class ClusterDetail(ClusterSummary):
    metadata: dict = Field(default_factory=dict)


class ClusterMember(BaseModel):
    memory_id: UUID
    content: str
    tags: list[str]
    type: str
    created_at: datetime
    distances: dict = Field(default_factory=dict)


class ClusterMembersResponse(BaseModel):
    cluster_id: UUID
    members: list[ClusterMember]
    total: int
    cursor_next: Optional[str] = None


class ClusterSourcesResponse(BaseModel):
    cluster_id: UUID
    sources: list[ClusterSummary]
    parent_clusters: list[ClusterSummary]


class NarrateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    narrative: str = Field(..., min_length=1, max_length=5000)


class ClusterStatusBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = Field(..., pattern="^(active|rejected|superseded)$")
    reason: Optional[str] = Field(None, max_length=500)


class ClustersListResponse(BaseModel):
    items: list[ClusterSummary]
    total: int
    cursor_next: Optional[str] = None


class ClusterStatsResponse(BaseModel):
    total_by_level: dict[str, int]
    total_by_status: dict[str, int]
    pending_narration: int
    graph_led_pct: Optional[float] = None
    last_run: Optional[datetime] = None
    avg_cluster_size: Optional[float] = None


class ClusterSearchResult(BaseModel):
    id: UUID
    level: str
    label: str
    narrative_preview: Optional[str] = None
    agent_identifier: Optional[str] = None
    period_start: date
    period_end: date
    member_count: int
    vector_score: float
    bm25_score: float


class ClusterSearchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_text: str = Field(..., min_length=3, max_length=2000)
    agent_identifier: Optional[str] = None
    level: Optional[str] = Field(None, pattern="^(weekly|monthly|quarterly|yearly)$")
    status: str = Field("active", pattern="^(active|candidate|rejected|superseded)$")
    limit: int = Field(10, ge=1, le=50)


class ClusterSearchResponse(BaseModel):
    results: list[ClusterSearchResult]
    count: int
    duration_ms: float


class ClusterNarrativeSummary(BaseModel):
    id: UUID
    label: str
    narrative: Optional[str] = None
    period_start: date
    period_end: date
    member_count: int
    source_count: int


class TelescopicViewResponse(BaseModel):
    # Boot order — oldest/broadest first: yearly → quarterly → monthly → weekly → recent_days.
    agent_identifier: str
    yearly: list[ClusterNarrativeSummary] = Field(default_factory=list)
    quarterly: list[ClusterNarrativeSummary] = Field(default_factory=list)
    monthly: list[ClusterNarrativeSummary] = Field(default_factory=list)
    weekly: list[ClusterNarrativeSummary] = Field(default_factory=list)
    recent_days: list[dict] = Field(default_factory=list, description="Raw memories from the last 3 complete days, oldest-first")


class FractalZoomBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_identifier: str = Field(..., min_length=1)
    cluster_id: Optional[UUID] = None
    query_text: Optional[str] = Field(None, min_length=3, max_length=2000)
    level: Optional[str] = Field(None, pattern="^(weekly|monthly|quarterly|yearly)$")
    limit: int = Field(20, ge=1, le=100)


class FractalZoomCluster(BaseModel):
    id: UUID
    level: str
    label: str
    narrative: Optional[str] = None
    period_start: date
    period_end: date
    member_count: int
    source_count: int
    score: Optional[float] = None


class FractalZoomMemory(BaseModel):
    memory_id: UUID
    type: str
    content: str
    weight: float
    tags: list[str]
    created_at: datetime
    score: Optional[float] = None


class FractalZoomResponse(BaseModel):
    agent_identifier: str
    parent: Optional[ClusterSummary] = None
    child_type: str  # "clusters" | "memories"
    clusters: list[FractalZoomCluster] = Field(default_factory=list)
    memories: list[FractalZoomMemory] = Field(default_factory=list)
    count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_jsonb(val):
    if val is None:
        return {}
    if isinstance(val, str):
        import json
        return json.loads(val)
    return val


def _cluster_summary(row) -> ClusterSummary:
    return ClusterSummary(
        id=row["id"],
        agent_id=row["agent_id"],
        level=row["level"],
        label=row["label"],
        detail=row.get("detail"),
        narrative=row.get("narrative"),
        member_count=row["member_count"],
        source_count=row["source_count"],
        pattern_flags=_parse_jsonb(row.get("pattern_flags")),
        period_start=row["period_start"],
        period_end=row["period_end"],
        status=row["status"],
        narrated_at=row.get("narrated_at"),
        created_at=row["created_at"],
    )


def _cluster_detail(row) -> ClusterDetail:
    base = _cluster_summary(row).model_dump()
    return ClusterDetail(**base, metadata=_parse_jsonb(row.get("metadata")))


# ---------------------------------------------------------------------------
# Endpoints — search + telescopic (Memory Agent v1.3)
# ---------------------------------------------------------------------------

@router.post("/search")
async def search_clusters(
    body: ClusterSearchBody,
    actor: dict = Depends(get_current_user),
) -> ClusterSearchResponse:
    import time
    t0 = time.time()
    pool = await get_pool()

    query_embedding = await embed_text(body.query_text, prompt_name="query")

    async with pool.acquire() as conn:
        # $1 = embedding vector, $2 = query text (BM25), $3 = status
        conditions = ["mc.status = $3"]
        params: list = [query_embedding, body.query_text, body.status]

        # Contamination prevention — Pepe's "mesa" rule (Spec §2):
        #   with agent_identifier -> that agent's own clusters + SIN_AUTOR
        #   without, non-super     -> only SIN_AUTOR (generic/technical)
        #   without, super         -> no agent filter (sees all)
        if body.agent_identifier:
            agent = await resolve_agent_for_actor(conn, actor, body.agent_identifier)
            params.append(agent["identifier"])
            conditions.append(
                f"(a.identifier = ${len(params)} OR a.identifier = 'SIN_AUTOR')")
        elif not actor.get("is_super"):
            conditions.append("a.identifier = 'SIN_AUTOR'")

        if body.level:
            params.append(body.level)
            conditions.append(f"mc.level = ${len(params)}")

        params.append(body.limit)
        where = " AND ".join(conditions)

        rows = await conn.fetch(f"""
            SELECT mc.id, mc.level, mc.label, mc.narrative,
                   a.identifier AS agent_identifier,
                   mc.period_start, mc.period_end,
                   coalesce(array_length(mc.member_ids, 1), 0) AS member_count,
                   1 - (mc.centroid <=> $1::vector) AS vector_score,
                   ts_rank(to_tsvector('spanish', mc.label),
                           plainto_tsquery('spanish', $2)) AS bm25_score
            FROM memory_clusters mc
            JOIN agents a ON a.id = mc.agent_id
            WHERE {where}
              AND mc.centroid IS NOT NULL
              AND (1 - (mc.centroid <=> $1::vector) > 0.3
                   OR ts_rank(to_tsvector('spanish', mc.label),
                              plainto_tsquery('spanish', $2)) > 0.05)
            ORDER BY (1 - (mc.centroid <=> $1::vector)) DESC
            LIMIT ${len(params)}
        """, *params)

    results = []
    for r in rows:
        narrative = r["narrative"]
        results.append(ClusterSearchResult(
            id=r["id"],
            level=r["level"],
            label=r["label"],
            narrative_preview=narrative[:200] if narrative else None,
            agent_identifier=r["agent_identifier"],
            period_start=r["period_start"],
            period_end=r["period_end"],
            member_count=r["member_count"],
            vector_score=float(r["vector_score"]),
            bm25_score=float(r["bm25_score"]),
        ))

    duration = (time.time() - t0) * 1000
    return ClusterSearchResponse(results=results, count=len(results), duration_ms=round(duration, 1))


@router.get("/telescopic")
async def get_telescopic_view(
    agent_identifier: str = Query(..., min_length=1),
    levels: str = Query("weekly,monthly,quarterly,yearly"),
    actor: dict = Depends(get_current_user),
) -> TelescopicViewResponse:
    requested_levels = [lv.strip() for lv in levels.split(",") if lv.strip()]
    valid_levels = {"weekly", "monthly", "quarterly", "yearly"}
    for lv in requested_levels:
        if lv not in valid_levels:
            raise HTTPException(422, f"invalid level: {lv}")

    limits = {"weekly": 4, "monthly": 3, "quarterly": 4, "yearly": 100}
    pool = await get_pool()

    async with pool.acquire() as conn:
        agent = await resolve_agent_for_actor(conn, actor, agent_identifier)
        aid = agent["id"]

        result: dict[str, list] = {}
        for lv in requested_levels:
            lim = limits.get(lv, 4)
            # Take the N most-recent clusters, then return them OLDEST-FIRST so the
            # boot reads chronologically (yearly→quarterly→monthly→weekly = old→recent).
            rows = await conn.fetch("""
                SELECT id, label, narrative, period_start, period_end,
                       member_count, source_count FROM (
                    SELECT mc.id, mc.label, mc.narrative,
                           mc.period_start, mc.period_end,
                           coalesce(array_length(mc.member_ids, 1), 0) AS member_count,
                           coalesce(array_length(mc.source_ids, 1), 0) AS source_count
                    FROM memory_clusters mc
                    WHERE mc.agent_id = $1 AND mc.level = $2 AND mc.status = 'active'
                    ORDER BY mc.period_end DESC
                    LIMIT $3
                ) sub
                ORDER BY period_end ASC
            """, aid, lv, lim)
            result[lv] = [ClusterNarrativeSummary(
                id=r["id"], label=r["label"], narrative=r["narrative"],
                period_start=r["period_start"], period_end=r["period_end"],
                member_count=r["member_count"], source_count=r["source_count"],
            ) for r in rows]

        # Last 3 complete days of raw memories (finest, most-recent layer of the boot).
        recent_rows = await conn.fetch("""
            SELECT id, type, content, weight, tags, created_at
            FROM memories
            WHERE agent_id = $1 AND created_at >= (CURRENT_DATE - INTERVAL '3 days')
            ORDER BY created_at ASC
        """, aid)
        recent_days = [{
            "id": str(r["id"]), "type": r["type"],
            "content": r["content"], "weight": float(r["weight"]),
            "tags": list(r["tags"]), "created_at": r["created_at"].isoformat(),
        } for r in recent_rows]

    return TelescopicViewResponse(
        agent_identifier=agent_identifier,
        yearly=result.get("yearly", []),
        quarterly=result.get("quarterly", []),
        monthly=result.get("monthly", []),
        weekly=result.get("weekly", []),
        recent_days=recent_days,
    )


# Levels that can absorb each level into a broader period (progressive zoom).
_HIGHER_LEVELS = {
    "weekly": ["monthly", "quarterly", "yearly"],
    "monthly": ["quarterly", "yearly"],
    "quarterly": ["yearly"],
    "yearly": [],
}

# Safety caps per level — number of DISTINCT PERIODS kept (clusters are
# thematic: one week can hold 10+ clusters, so capping by cluster count
# would silently drop whole weeks). Only bite when consolidation lags badly.
_PROGRESSIVE_CAPS = {"weekly": 6, "monthly": 12, "quarterly": 8, "yearly": 50}


_PROGRESSIVE_SECTIONS = ("yearly", "quarterly", "monthly", "weekly", "recent_days")


@router.get("/telescopic/progressive")
async def get_progressive_view(
    agent_identifier: str = Query(..., min_length=1),
    max_recent_days: int = Query(14, ge=1, le=31),
    sections: str = Query("all"),
    actor: dict = Depends(get_current_user),
) -> TelescopicViewResponse:
    """Progressive-zoom telescopic view (Pepe's vision, day 101).

    Each temporal layer is the COMPRESSION of the previous one — closed
    periods are never re-read at finer granularity:
      - yearly: all active yearly clusters (previous years)
      - quarterly: only those NOT period-covered by an active yearly
      - monthly: only those NOT covered by an active quarterly/yearly
      - weekly: only those NOT covered by an active monthly or higher
      - recent_days: raw memories not yet woven into any active weekly
        cluster (capped at max_recent_days)

    Boot reads oldest→newest: compressed narrative for the distant past,
    full granularity only for the open period.

    `sections` selects which layers to compute ("all" or a comma list of
    yearly,quarterly,monthly,weekly,recent_days) so clients with a
    per-response size limit (e.g. MCP output caps) can load the view in
    chronological chapters across 2-3 calls.
    """
    if sections == "all":
        wanted = set(_PROGRESSIVE_SECTIONS)
    else:
        wanted = {s.strip() for s in sections.split(",") if s.strip()}
        invalid = wanted - set(_PROGRESSIVE_SECTIONS)
        if invalid:
            raise HTTPException(422, f"invalid sections: {sorted(invalid)}")

    pool = await get_pool()
    async with pool.acquire() as conn:
        agent = await resolve_agent_for_actor(conn, actor, agent_identifier)
        aid = agent["id"]

        result: dict[str, list] = {lv: [] for lv in
                                   ("yearly", "quarterly", "monthly", "weekly")}
        for lv in ("yearly", "quarterly", "monthly", "weekly"):
            if lv not in wanted:
                continue
            rows = await conn.fetch("""
                WITH unabsorbed AS (
                    SELECT mc.id, mc.label, mc.narrative,
                           mc.period_start, mc.period_end,
                           coalesce(array_length(mc.member_ids, 1), 0) AS member_count,
                           coalesce(array_length(mc.source_ids, 1), 0) AS source_count
                    FROM memory_clusters mc
                    WHERE mc.agent_id = $1 AND mc.level = $2 AND mc.status = 'active'
                      -- period absorption: a higher-level active cluster
                      -- covering the period compresses it
                      AND NOT EXISTS (
                          SELECT 1 FROM memory_clusters p
                          WHERE p.agent_id = $1 AND p.status = 'active'
                            AND p.level = ANY($3::text[])
                            AND p.period_start <= mc.period_start
                            AND p.period_end >= mc.period_end)
                      -- lineage absorption: a thematic weekly under a week
                      -- rollup is read through its rollup, never raw
                      AND NOT EXISTS (
                          SELECT 1 FROM memory_clusters r
                          WHERE r.agent_id = $1 AND r.status = 'active'
                            AND mc.id = ANY(r.source_ids))
                )
                SELECT * FROM unabsorbed
                WHERE period_end IN (
                    SELECT DISTINCT period_end FROM unabsorbed
                    ORDER BY period_end DESC LIMIT $4)
                ORDER BY period_end ASC, period_start ASC, id ASC
            """, aid, lv, _HIGHER_LEVELS[lv], _PROGRESSIVE_CAPS[lv])
            result[lv] = [ClusterNarrativeSummary(
                id=r["id"], label=r["label"], narrative=r["narrative"],
                period_start=r["period_start"], period_end=r["period_end"],
                member_count=r["member_count"], source_count=r["source_count"],
            ) for r in rows]

        # Loose days = ONLY the open edge. A memory is hidden if its week is
        # already closed (its date falls inside ANY active weekly period) OR
        # it was woven into a weekly cluster. Closed periods are never
        # re-read — unclustered outliers of a consolidated week included
        # (Eco's report, day 102). Capped so a long consolidation gap can't
        # flood the boot context.
        recent_rows = []
        if "recent_days" in wanted:
            recent_rows = await conn.fetch("""
                SELECT m.id, m.type, m.content, m.weight, m.tags, m.created_at
                FROM memories m
                WHERE m.agent_id = $1
                  AND m.created_at >= CURRENT_DATE - $2::int
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_clusters mc
                      WHERE mc.agent_id = $1 AND mc.level = 'weekly'
                        AND mc.status = 'active'
                        AND (m.created_at::date
                                 BETWEEN mc.period_start AND mc.period_end
                             OR m.id = ANY(mc.member_ids)))
                ORDER BY m.created_at ASC
            """, aid, max_recent_days)
        recent_days = [{
            "id": str(r["id"]), "type": r["type"],
            "content": r["content"], "weight": float(r["weight"]),
            "tags": list(r["tags"]), "created_at": r["created_at"].isoformat(),
        } for r in recent_rows]

    return TelescopicViewResponse(
        agent_identifier=agent_identifier,
        yearly=result["yearly"],
        quarterly=result["quarterly"],
        monthly=result["monthly"],
        weekly=result["weekly"],
        recent_days=recent_days,
    )


_LEVEL_RANK = "CASE level WHEN 'yearly' THEN 0 WHEN 'quarterly' THEN 1 WHEN 'monthly' THEN 2 ELSE 3 END"


@router.post("/zoom")
async def fractal_zoom(
    body: FractalZoomBody,
    actor: dict = Depends(get_current_user),
) -> FractalZoomResponse:
    """Fractal drill-down over the cluster hierarchy.

    Without cluster_id: entry at the highest abstraction level available
    for the agent (yearly → quarterly → monthly → weekly), or `level` if
    given. With cluster_id: returns that cluster's children — source
    clusters for higher levels, member memories for weekly. Optional
    query_text ranks children semantically within the scope; otherwise
    children come in chronological order. Each child carries its id so
    the caller can keep zooming in.
    """
    pool = await get_pool()
    q_emb = None
    if body.query_text:
        q_emb = await embed_text(body.query_text, prompt_name="query")

    async with pool.acquire() as conn:
        agent = await resolve_agent_for_actor(conn, actor, body.agent_identifier)
        aid = agent["id"]

        parent = None
        cluster_children: list = []
        memory_children: list = []
        child_type = "clusters"

        if body.cluster_id is None:
            entry_level = body.level
            if entry_level is None:
                entry_level = await conn.fetchval(f"""
                    SELECT level FROM memory_clusters
                    WHERE agent_id = $1 AND status = 'active'
                    ORDER BY {_LEVEL_RANK}
                    LIMIT 1
                """, aid)
            if entry_level is not None:
                # Entry hides lineage-absorbed clusters (thematic weeklies
                # under a rollup) — you reach them by zooming the rollup.
                cluster_children = await _zoom_clusters(
                    conn,
                    "mc.agent_id = $1 AND mc.level = $2 AND mc.status = 'active'"
                    " AND NOT EXISTS (SELECT 1 FROM memory_clusters r"
                    " WHERE r.agent_id = $1 AND r.status = 'active'"
                    " AND mc.id = ANY(r.source_ids))",
                    [aid, entry_level], q_emb, body.limit)
        else:
            cluster = await resolve_cluster_for_actor(conn, actor, body.cluster_id)
            if cluster["agent_id"] != aid:
                raise HTTPException(404)
            cluster["member_count"] = len(cluster["member_ids"])
            cluster["source_count"] = len(cluster["source_ids"] or [])
            parent = _cluster_summary(cluster)
            if cluster["source_ids"]:
                cluster_children = await _zoom_clusters(
                    conn, "mc.id = ANY($1::uuid[]) AND mc.agent_id = $2",
                    [cluster["source_ids"], aid], q_emb, body.limit)
            else:
                child_type = "memories"
                memory_children = await _zoom_memories(
                    conn, actor, cluster["member_ids"], q_emb, body.limit)

    return FractalZoomResponse(
        agent_identifier=body.agent_identifier,
        parent=parent,
        child_type=child_type,
        clusters=cluster_children,
        memories=memory_children,
        count=len(cluster_children) + len(memory_children),
    )


async def _zoom_clusters(conn, where: str, params: list,
                         q_emb, limit: int) -> list[FractalZoomCluster]:
    if q_emb is not None:
        params = [*params, q_emb, limit]
        score_sql = f"1 - (mc.centroid <=> ${len(params) - 1}::vector)"
        order_sql = f"(mc.centroid IS NULL), mc.centroid <=> ${len(params) - 1}::vector"
    else:
        params = [*params, limit]
        score_sql = "NULL::float"
        order_sql = "mc.period_end ASC"
    rows = await conn.fetch(f"""
        SELECT mc.id, mc.level, mc.label, mc.narrative,
               mc.period_start, mc.period_end,
               coalesce(array_length(mc.member_ids, 1), 0) AS member_count,
               coalesce(array_length(mc.source_ids, 1), 0) AS source_count,
               {score_sql} AS score
        FROM memory_clusters mc
        WHERE {where}
        ORDER BY {order_sql}
        LIMIT ${len(params)}
    """, *params)
    return [FractalZoomCluster(
        id=r["id"], level=r["level"], label=r["label"], narrative=r["narrative"],
        period_start=r["period_start"], period_end=r["period_end"],
        member_count=r["member_count"], source_count=r["source_count"],
        score=float(r["score"]) if r["score"] is not None else None,
    ) for r in rows]


async def _zoom_memories(conn, actor: dict, member_ids: list,
                         q_emb, limit: int) -> list[FractalZoomMemory]:
    if q_emb is not None:
        rows = await conn.fetch("""
            SELECT *, 1 - (embedding <=> $2::vector) AS score
            FROM memories WHERE id = ANY($1::uuid[])
            ORDER BY (embedding IS NULL), embedding <=> $2::vector
        """, member_ids, q_emb)
    else:
        rows = await conn.fetch("""
            SELECT *, NULL::float AS score
            FROM memories WHERE id = ANY($1::uuid[])
            ORDER BY created_at ASC
        """, member_ids)
    vis = await precompute_read_visibility(conn, actor)
    out: list[FractalZoomMemory] = []
    for mem in rows:
        if not check_read_memory(vis, mem):
            continue
        out.append(FractalZoomMemory(
            memory_id=mem["id"], type=mem["type"], content=mem["content"],
            weight=float(mem["weight"]), tags=list(mem["tags"]),
            created_at=mem["created_at"],
            score=float(mem["score"]) if mem["score"] is not None else None,
        ))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Endpoints — CRUD (existing v2.0)
# ---------------------------------------------------------------------------

@router.get("")
async def list_clusters(
    agent_identifier: str = Query(..., min_length=1),
    level: Optional[str] = Query(None, pattern="^(weekly|monthly|quarterly|yearly)$"),
    status: Optional[str] = Query(None, pattern="^(candidate|active|rejected|superseded)$"),
    pending_narration: bool = Query(False),
    pattern_flags: Optional[str] = Query(None),
    period_start: Optional[date] = Query(None),
    period_end: Optional[date] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    actor: dict = Depends(get_current_user),
) -> ClustersListResponse:
    pool = await get_pool()
    async with pool.acquire() as conn:
        agent = await resolve_agent_for_actor(conn, actor, agent_identifier)
        conditions = ["mc.agent_id = $1"]
        params: list = [agent["id"]]
        if level:
            params.append(level)
            conditions.append(f"mc.level = ${len(params)}")
        if status:
            params.append(status)
            conditions.append(f"mc.status = ${len(params)}")
        if pending_narration:
            conditions.append("mc.narrative IS NULL AND mc.status = 'candidate'")
        if pattern_flags:
            import json as _json
            try:
                _json.loads(pattern_flags)
            except (_json.JSONDecodeError, TypeError):
                raise HTTPException(422, "pattern_flags must be valid JSON")
            params.append(pattern_flags)
            conditions.append(f"mc.pattern_flags @> ${len(params)}::jsonb")
        if period_start:
            params.append(period_start)
            conditions.append(f"mc.period_start >= ${len(params)}")
        if period_end:
            params.append(period_end)
            conditions.append(f"mc.period_end <= ${len(params)}")
        where = " AND ".join(conditions)
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM memory_clusters mc WHERE {where}", *params)
        items, next_cursor = await paginate(conn, f"""
            SELECT mc.*, array_length(member_ids, 1) AS member_count,
                   coalesce(array_length(source_ids, 1), 0) AS source_count
            FROM memory_clusters mc WHERE {where}
        """, list(params), limit, cursor)
    return {"items": [_cluster_summary(r) for r in items], "total": total, "cursor_next": next_cursor}


# /stats BEFORE /{cluster_id} — otherwise FastAPI matches "stats" as UUID
@router.get("/stats")
async def cluster_stats(
    agent_identifier: str = Query(..., min_length=1),
    actor: dict = Depends(get_current_user),
) -> ClusterStatsResponse:
    pool = await get_pool()
    async with pool.acquire() as conn:
        agent = await resolve_agent_for_actor(conn, actor, agent_identifier)
        aid = agent["id"]
        by_level = await conn.fetch(
            "SELECT level, COUNT(*) AS cnt FROM memory_clusters WHERE agent_id=$1 GROUP BY level", aid)
        by_status = await conn.fetch(
            "SELECT status, COUNT(*) AS cnt FROM memory_clusters WHERE agent_id=$1 GROUP BY status", aid)
        pending = await conn.fetchval(
            "SELECT COUNT(*) FROM memory_clusters WHERE agent_id=$1 AND narrative IS NULL AND status='candidate'", aid)
        avg_size = await conn.fetchval(
            "SELECT AVG(array_length(member_ids,1)) FROM memory_clusters WHERE agent_id=$1 AND status='active'", aid)
        last_run_row = await conn.fetchrow("""
            SELECT finished_at, metrics FROM cell_runs
            WHERE agent_id=$1 AND cell_type='consolidation' AND status='completed'
            ORDER BY finished_at DESC LIMIT 1
        """, aid)
        _raw_m = last_run_row["metrics"] if last_run_row else {}
        if isinstance(_raw_m, str):
            import json as _json
            _raw_m = _json.loads(_raw_m)
        graph_led = (_raw_m or {}).get("graph_led_pct")
        last_run = last_run_row["finished_at"] if last_run_row else None
    return {
        "total_by_level": {r["level"]: r["cnt"] for r in by_level},
        "total_by_status": {r["status"]: r["cnt"] for r in by_status},
        "pending_narration": pending,
        "graph_led_pct": graph_led,
        "last_run": last_run,
        "avg_cluster_size": float(avg_size) if avg_size else None,
    }


@router.get("/{cluster_id}")
async def get_cluster(cluster_id: UUID, actor: dict = Depends(get_current_user)) -> ClusterDetail:
    pool = await get_pool()
    async with pool.acquire() as conn:
        cluster = await resolve_cluster_for_actor(conn, actor, cluster_id)
        cluster["member_count"] = len(cluster["member_ids"])
        cluster["source_count"] = len(cluster["source_ids"] or [])
    return _cluster_detail(cluster)


@router.get("/{cluster_id}/members")
async def get_cluster_members(
    cluster_id: UUID, limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    actor: dict = Depends(get_current_user),
) -> ClusterMembersResponse:
    pool = await get_pool()
    async with pool.acquire() as conn:
        cluster = await resolve_cluster_for_actor(conn, actor, cluster_id)
        all_ids = cluster["member_ids"]
        rows = await conn.fetch(
            "SELECT * FROM memories WHERE id = ANY($1::uuid[]) ORDER BY created_at DESC",
            all_ids)
        md = _parse_jsonb(cluster.get("metadata")).get("member_distances", {})
        vis = await precompute_read_visibility(conn, actor)
        visible = []
        for mem in rows:
            if check_read_memory(vis, mem):
                visible.append(ClusterMember(
                    memory_id=mem["id"],
                    content=mem["content"],
                    tags=list(mem["tags"]),
                    type=mem["type"],
                    created_at=mem["created_at"],
                    distances=md.get(str(mem["id"]), {}),
                ))
        if cursor:
            try:
                cursor_dt = datetime.fromisoformat(cursor)
            except (ValueError, TypeError):
                raise HTTPException(422, "invalid cursor format")
            visible = [m for m in visible if m.created_at < cursor_dt]
        page = visible[:limit]
        next_cursor = page[-1].created_at.isoformat() if len(visible) > limit else None
    return ClusterMembersResponse(
        cluster_id=cluster_id, members=page,
        total=len(visible), cursor_next=next_cursor)


@router.get("/{cluster_id}/sources")
async def get_cluster_sources(
    cluster_id: UUID, actor: dict = Depends(get_current_user),
) -> ClusterSourcesResponse:
    pool = await get_pool()
    async with pool.acquire() as conn:
        cluster = await resolve_cluster_for_actor(conn, actor, cluster_id)
        agent_id = cluster["agent_id"]
        sources = []
        if cluster["source_ids"]:
            source_rows = await conn.fetch(
                "SELECT *, coalesce(array_length(member_ids,1),0) AS member_count, "
                "coalesce(array_length(source_ids,1),0) AS source_count "
                "FROM memory_clusters WHERE id = ANY($1::uuid[]) AND agent_id = $2",
                cluster["source_ids"], agent_id)
            sources = [_cluster_summary(r) for r in source_rows]
        parents = await conn.fetch("""
            SELECT *, coalesce(array_length(member_ids,1),0) AS member_count,
                   coalesce(array_length(source_ids,1),0) AS source_count
            FROM memory_clusters WHERE $1::uuid = ANY(source_ids) AND agent_id = $2
        """, cluster_id, agent_id)
    return {"cluster_id": str(cluster_id),
            "sources": sources,
            "parent_clusters": [_cluster_summary(r) for r in parents]}


@router.put("/{cluster_id}/narrate")
async def narrate_cluster(
    cluster_id: UUID, body: NarrateBody,
    actor: dict = Depends(get_current_user),
) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        cluster = await resolve_cluster_for_actor(conn, actor, cluster_id)
        await conn.execute(
            "UPDATE memory_clusters SET narrative=$1, narrated_at=NOW() WHERE id=$2",
            body.narrative, cluster_id)
        org_id = await conn.fetchval(
            "SELECT organization_id FROM workspaces WHERE id=$1",
            cluster["workspace_id"])
        await broadcast_event("cluster.narrated",
            {"cluster_id": str(cluster_id),
             "agent_identifier": cluster["agent_identifier"]},
            org_id=org_id)
    return {"ok": True}


@router.put("/{cluster_id}/status")
async def update_cluster_status(
    cluster_id: UUID, body: ClusterStatusBody,
    actor: dict = Depends(get_current_user),
) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        cluster = await resolve_cluster_for_actor(conn, actor, cluster_id)
        valid = {("candidate", "active"), ("candidate", "rejected"), ("active", "superseded")}
        if (cluster["status"], body.status) not in valid:
            raise HTTPException(422, f"invalid transition {cluster['status']} -> {body.status}")
        await conn.execute(
            "UPDATE memory_clusters SET status=$1 WHERE id=$2",
            body.status, cluster_id)
        event_map = {"active": "cluster.promoted", "rejected": "cluster.rejected",
                     "superseded": "cluster.superseded"}
        org_id = await conn.fetchval(
            "SELECT organization_id FROM workspaces WHERE id=$1",
            cluster["workspace_id"])
        event_data = {"cluster_id": str(cluster_id),
             "agent_identifier": cluster["agent_identifier"]}
        if body.reason:
            event_data["reason"] = body.reason
        await broadcast_event(event_map[body.status], event_data, org_id=org_id)
    return {"ok": True}


@router.delete("/{cluster_id}", status_code=204)
async def delete_cluster(cluster_id: UUID, actor: dict = Depends(get_current_user)):
    if not actor.get("is_super"):
        raise HTTPException(403, "cluster delete requires super access")
    pool = await get_pool()
    async with pool.acquire() as conn:
        if not await conn.fetchval("SELECT 1 FROM memory_clusters WHERE id=$1", cluster_id):
            raise HTTPException(404)
        await conn.execute(
            "UPDATE memory_clusters SET status='rejected' WHERE id=$1", cluster_id)
