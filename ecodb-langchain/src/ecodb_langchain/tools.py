"""LangChain tools backed by EcoDB.

Parity with the agentic surface of the EcoDB MCP server: search, recent,
save, read, graph navigation (neighbours / path / fuzzy node search /
status) and triple writes. These are the tools an agent reasons with — the
document-ingestion and admin MCP tools are intentionally excluded (an agent
rarely needs them; add them with ``extra_tools`` if you do).

``make_ecodb_tools(client)`` returns a list of ``BaseTool`` bound to a client,
ready to hand to ``llm.bind_tools(...)`` or a LangGraph ``ToolNode``.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import BaseTool, tool

from .client import EcoDBClient


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _trim_results(data: dict, max_chars: int = 600) -> list[dict]:
    """Compact GAMR results for an LLM context window (drop embeddings/raw blobs)."""
    out = []
    for r in data.get("results", []):
        out.append(
            {
                "id": r.get("id"),
                "type": r.get("type"),
                "score": round(r.get("score", 0), 4) if isinstance(r.get("score"), (int, float)) else r.get("score"),
                "source_type": r.get("source_type", "memory"),
                "content": (r.get("content") or "")[:max_chars],
                "tags": r.get("tags", []),
                "created_at": r.get("created_at"),
            }
        )
    return out


def make_ecodb_tools(client: EcoDBClient) -> list[BaseTool]:
    """Build the EcoDB LangChain toolset bound to ``client``."""

    @tool
    def ecodb_search(
        query_text: str,
        limit: int = 8,
        query_type: Optional[str] = None,
        type: Optional[str] = None,
        graph_discovery: bool = False,
    ) -> str:
        """Search EcoDB long-term memory with the GAMR engine (semantic + knowledge graph +
        temporal + contradiction detection). Use this to recall facts, decisions, past work
        or context before answering. ``query_type`` may be factual|historical|analytical|contextual.
        ``type`` filters by memory type. Returns ranked results with scores."""
        data = client.search(
            query_text,
            limit=limit,
            query_type=query_type,
            type=type,
            graph_discovery=graph_discovery,
        )
        return _dumps({"results": _trim_results(data), "query_type": data.get("query_type")})

    @tool
    def ecodb_search_recent(limit: int = 10, type: Optional[str] = None) -> str:
        """List the most recent memories (newest first), filtered by the actor's permissions.
        Use when the user asks 'what have we done lately' or for a time-ordered view rather
        than a semantic query."""
        data = client.search_recent(limit=limit)
        items = data.get("items", data.get("results", []))
        if type:
            items = [m for m in items if m.get("type") == type]
        trimmed = [
            {"id": m.get("id"), "type": m.get("type"), "content": (m.get("content") or "")[:600],
             "tags": m.get("tags", []), "created_at": m.get("created_at")}
            for m in items
        ]
        return _dumps({"items": trimmed})

    @tool
    def ecodb_save_memory(content: str, type: str, tags: Optional[list[str]] = None) -> str:
        """Persist a new memory in EcoDB so it survives across sessions. ``type`` is REQUIRED, one of
        momento, decision, acuerdo, tecnico, descubrimiento, observacion, referencia, caso, skill.
        Use for durable facts, decisions or discoveries — not for ephemeral chit-chat.
        Returns the created memory id."""
        data = client.save_memory(content, type=type, tags=tags)
        return _dumps({"status": "ok", "id": data.get("id"), "type": data.get("type"),
                       "weight": data.get("weight"), "created_at": data.get("created_at")})

    @tool
    def ecodb_read_memory(memory_id: str) -> str:
        """Read a single memory by its UUID (full content + metadata). Use to expand a result
        returned by ecodb_search when you need the complete text."""
        data = client.read_memory(memory_id)
        return _dumps(data)

    @tool
    def ecodb_graph_neighbors(node: str, depth: int = 1) -> str:
        """Explore the knowledge graph: return entities connected to ``node`` within ``depth``
        hops (1-3). Use to discover how a concept/person/project relates to others."""
        return _dumps(client.neighbors(node, depth=depth))

    @tool
    def ecodb_graph_path(source: str, target: str, max_depth: int = 6) -> str:
        """Find the shortest path between two entities in the knowledge graph. Use to explain
        how two things are connected."""
        return _dumps(client.path_between(source, target, max_depth=max_depth))

    @tool
    def ecodb_search_nodes(query: str, limit: int = 10) -> str:
        """Fuzzy-search graph node names (min 3 chars) when you don't remember the exact entity
        name. Call this before ecodb_graph_neighbors / ecodb_graph_path to resolve a name."""
        return _dumps(client.search_nodes(query, limit=limit))

    @tool
    def ecodb_graph_status() -> str:
        """Return knowledge-graph statistics (node/edge counts, etc.)."""
        return _dumps(client.graph_status())

    @tool
    def ecodb_save_triple(subject: str, predicate: str, object: str) -> str:
        """Add a subject-predicate-object fact to the knowledge graph (e.g. 'EcoDB' 'uses'
        'pgvector'). Use to record an explicit relationship between two entities."""
        return _dumps(client.save_triple(subject, predicate, object))

    @tool
    def ecodb_search_clusters(query_text: str, agent_identifier: Optional[str] = None,
                              level: Optional[str] = None, limit: int = 10) -> str:
        """Search consolidated memory clusters (fractal long-term memory) by semantic similarity.
        Clusters are weekly/monthly/quarterly/yearly narrative consolidations. Without
        ``agent_identifier`` returns only generic/technical (SIN_AUTOR) clusters. ``level`` filters
        by temporal level. Use to recall higher-level patterns rather than individual memories."""
        return _dumps(client.search_clusters(query_text, agent_identifier=agent_identifier,
                                             level=level, limit=limit))

    @tool
    def ecodb_telescopic_view(agent_identifier: str,
                              levels: str = "weekly,monthly,quarterly,yearly") -> str:
        """Load an agent's full fractal memory chain for boot/context: weekly→monthly→quarterly→yearly
        narrative consolidations stacked telescopically. Use at session start to load long-term
        self-context, or to understand an agent's accumulated history at a glance."""
        return _dumps(client.get_telescopic_view(agent_identifier, levels=levels))

    @tool
    def ecodb_briefing(agent_identifier: str) -> str:
        """Get an agent's briefing: active foresights (upcoming/predicted events), open identity
        tensions (declared vs observed traits), and a telescopic memory summary. Use to answer
        'what's coming up' or 'what should I be aware of' for an agent."""
        return _dumps(client.get_briefing(agent_identifier))

    @tool
    def ecodb_read_cluster(cluster_id: str, include_members: bool = False) -> str:
        """Read a single memory cluster by UUID: full narrative + metadata. Set ``include_members``
        to also fetch the underlying memories. Use to expand a cluster from ecodb_search_clusters."""
        return _dumps(client.read_cluster(cluster_id, include_members=include_members))

    return [
        ecodb_search,
        ecodb_search_recent,
        ecodb_save_memory,
        ecodb_read_memory,
        ecodb_graph_neighbors,
        ecodb_graph_path,
        ecodb_search_nodes,
        ecodb_graph_status,
        ecodb_save_triple,
        ecodb_search_clusters,
        ecodb_telescopic_view,
        ecodb_briefing,
        ecodb_read_cluster,
    ]
