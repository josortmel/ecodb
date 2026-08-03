"""EcoDB MCP wrapper — .

MCP → HTTP API translator. No business logic. Each tool proxies to the
corresponding API endpoint and returns JSON.

Core tools (agent workflow — save, search, read, graph):
- save_memory       → POST /memories
- search            → POST /search       (GAMR Etapa 3)
- search_recent     → GET  /memories/recent
- read_memory       → GET  /memories/{id}
- save_triple       → POST /graph/triples
- neighbors         → GET  /graph/neighbors/{node}

Full 40-tool set covers all GAMR phases (memory, graph, documents, identity,
metacognition clusters + progressive/fractal navigation).

Auth:
- ECODB_API_KEY en env (formato `ecodb_<32-bytes-base64url>`).
- Al primer request, intercambio API key → JWT (POST /auth/token), cache.
- Si JWT expira (HTTP 401), refresh automatico una vez.

Configuracion:
- ECODB_API_URL    (default http://localhost:8080)
- ECODB_API_KEY    (obligatorio)
- ECODB_TIMEOUT    (default 60.0)

Run:
- Subprocess stdio: `python server.py` (configuracion para Claude Code MCP).
- Standalone test: importa mcp_call_X() directo.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import re as _re
from urllib.parse import quote, urlparse

import httpx

_URL_SCHEME_RE = _re.compile(r'^(https?|ftp|file|rtsp|rtmp)://', _re.IGNORECASE)
_UUID_RE = _re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', _re.I)
from mcp.server.fastmcp import FastMCP, Image


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

ECODB_API_URL = os.environ.get("ECODB_API_URL", "http://localhost:8080").rstrip("/")
ECODB_API_KEY = os.environ.get("ECODB_API_KEY", "")
ECODB_TIMEOUT = float(os.environ.get("ECODB_TIMEOUT", "60.0"))

# Transport selector — 
# - "stdio" (default, dev local): subprocess invocado por Claude Code.
# - "sse": HTTP server-sent events, util para containerizar y servir por red.
# - "streamable-http": variante mas moderna del transport HTTP.
# El docker-compose pasa MCP_TRANSPORT=sse para exponer el puerto 8091.
def _resolve_transport() -> str:
    import sys
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--transport" and i < len(sys.argv) - 1:
            return sys.argv[i + 1].lower()
        if arg.startswith("--transport="):
            return arg.split("=", 1)[1].lower()
    return os.environ.get("MCP_TRANSPORT", "stdio").lower()

MCP_TRANSPORT = _resolve_transport()
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.environ.get("MCP_PORT", "8091"))

if not ECODB_API_KEY:
    # No raise al import — permite ejecutar tests con env diferente. Pero
    # cualquier tool fallara con mensaje claro al primer call.
    print("[ecodb-mcp] WARNING: ECODB_API_KEY no configurada", file=sys.stderr)

# VS1-MCP fix .
# Permitimos https a cualquier host (cloud deployments legitimos) pero
# rechazamos http a hosts NO internos — http externo transmite ECODB_API_KEY
# en claro al destino, vector SSRF + exfiltracion. Hosts internos OK con http
# (red Docker, localhost, host.docker.internal son trusted dentro del entorno).
_INTERNAL_HOSTS = {"localhost", "127.0.0.1", "host.docker.internal", "ecodb-api", "api"}

# Media store: default is <project_root>/media/. Override with ECODB_MEDIA_DIR env var.
# Docker MCP sets ECODB_MEDIA_DIR=/app/media in docker-compose.yml.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_STORE_DIR = Path(os.environ.get("ECODB_MEDIA_DIR", str(_PROJECT_ROOT / "media"))).resolve()
MAX_MEDIA_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
_DENIED_EXTENSIONS = {".env", ".key", ".pem", ".pfx", ".p12", ".p8", ".ppk", ".der", ".yaml", ".toml", ".ini", ".config", ".exe", ".dll", ".bat", ".ps1", ".sh", ".cmd"}
_EMBEDDABLE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_NO_COPY_EXTENSIONS = {".html", ".css", ".htm"}

# Predicate normalization pipeline (Fase 3b gobernanza)
MAPPER_THRESHOLD = float(os.environ.get("MAPPER_THRESHOLD", "0.70"))
_predicate_cache: dict[tuple[str, str, str], str] = {}  # (lexema, subject_type, object_type) → canonical
_alias_cache: dict[tuple[str, str | None], str] = {}  # (alias, domain) → canonical
_alias_cache_loaded: bool = False


def _normalize_predicate_lexical(predicate: str) -> str:
    """Etapa 1: normalización léxica. lowercase, trim, espacios→underscore."""
    return predicate.strip().lower().replace(" ", "_").replace("-", "_")


def _resolve_predicate(predicate: str, subject_type: str = "unknown", object_type: str = "unknown") -> dict:
    """Pipeline completo de normalización de predicados (5 etapas).

    Returns dict con:
      canonical: str — predicado canónico (o original si no mapea)
      original: str — predicado tal como llegó
      confidence: float — mapper_confidence (1.0 si alias/cache hit, 0.0-1.0 si embedding)
      needs_review: bool — True si confidence < MAPPER_THRESHOLD
      method: str — qué etapa resolvió (lexical, alias, type_validated, embedding, pending)
    """
    original = predicate
    lexeme = _normalize_predicate_lexical(predicate)

    # Etapa 1+2: cache de lexemas + alias
    cache_key = (lexeme, subject_type, object_type)
    if cache_key in _predicate_cache:
        return {"canonical": _predicate_cache[cache_key], "original": original,
                "confidence": 1.0, "needs_review": False, "method": "cache"}

    # Domain-específico primero, genérico como fallback (BC3: coherente con SQL)
    for domain_key in [(lexeme, subject_type), (lexeme, object_type), (lexeme, None)]:
        if domain_key in _alias_cache:
            canonical = _alias_cache[domain_key]
            _predicate_cache[cache_key] = canonical
            return {"canonical": canonical, "original": original,
                    "confidence": 1.0, "needs_review": False, "method": "alias"}

    # Etapa 2b: exact match contra predicates_canonical
    try:
        data = _api_call("GET", "/graph/predicates/resolve", params={
            "predicate": lexeme,
            "subject_type": subject_type,
            "object_type": object_type,
        })
        if data.get("canonical"):
            canonical = data["canonical"]
            confidence = float(data.get("confidence", 0.0))
            method = data.get("method", "embedding")
            needs_review = confidence < MAPPER_THRESHOLD
            if confidence >= MAPPER_THRESHOLD:
                _predicate_cache[cache_key] = canonical
            return {"canonical": canonical, "original": original,
                    "confidence": confidence, "needs_review": needs_review, "method": method}
    except RuntimeError:
        pass

    # Etapa 5: no mapeó — va a pending
    return {"canonical": lexeme, "original": original,
            "confidence": 0.0, "needs_review": True, "method": "pending"}


def _load_alias_cache() -> None:
    """Carga alias de la BD al cache local. Se llama lazy al primer uso."""
    global _alias_cache, _alias_cache_loaded
    try:
        data = _api_call("GET", "/graph/predicates/aliases")
        for alias_entry in data.get("aliases", []):
            key = (alias_entry["alias"], alias_entry.get("domain") or None)
            _alias_cache[key] = alias_entry["canonical"]
    except RuntimeError:
        pass
    _alias_cache_loaded = True  # siempre True tras intento, incluso si vacío (BC4)


def _copy_to_media_store(source_path: str, memory_id: str) -> str:
    """Copia archivo a media store estable. Devuelve ruta de la copia."""
    real_path = Path(os.path.realpath(source_path))
    if not real_path.is_file():
        raise RuntimeError(f"file_path is not a regular file: {source_path}")
    size = real_path.stat().st_size
    if size > MAX_MEDIA_FILE_SIZE:
        raise RuntimeError(f"file exceeds max size ({size} > {MAX_MEDIA_FILE_SIZE} bytes)")
    month_dir = MEDIA_STORE_DIR / datetime.now(timezone.utc).strftime("%Y-%m")
    month_dir.mkdir(parents=True, exist_ok=True)
    dest = month_dir / f"{memory_id}_{real_path.name}"
    shutil.copy2(str(real_path), str(dest))
    return str(dest)
_parsed_api_url = urlparse(ECODB_API_URL)
if _parsed_api_url.scheme == "http" and _parsed_api_url.hostname not in _INTERNAL_HOSTS:
    raise RuntimeError(
        f"ECODB_API_URL con http:// solo permitido a hosts internos {_INTERNAL_HOSTS}. "
        f"Got: {_parsed_api_url.hostname!r}. Para hosts externos usar https://."
    )
if _parsed_api_url.scheme not in ("http", "https"):
    raise RuntimeError(
        f"ECODB_API_URL scheme must be http or https, got: {_parsed_api_url.scheme!r}"
    )


# ---------------------------------------------------------------------------
# JWT cache + auth helpers
# ---------------------------------------------------------------------------

_jwt: Optional[str] = None
_jwt_expires_at: float = 0.0  # epoch seconds


def _ensure_jwt(client: httpx.Client, force_refresh: bool = False) -> str:
    """Devuelve un JWT valido. Si no hay o esta cerca de expirar, intercambia
    la API key por uno nuevo.

    El JWT TTL del API es 3600s; refrescamos con margen de 60s para evitar
    races con expiracion durante un request en vuelo.
    """
    global _jwt, _jwt_expires_at
    now = time.time()
    if not force_refresh and _jwt and now < _jwt_expires_at - 60:
        return _jwt
    if not ECODB_API_KEY:
        raise RuntimeError("ECODB_API_KEY no configurada en env vars")
    r = client.post(f"{ECODB_API_URL}/auth/token", json={"api_key": ECODB_API_KEY})
    if r.status_code != 200:
        raise RuntimeError(f"auth/token failed: HTTP {r.status_code}")
    # BC1 fix 
    # error page con 200 OK), r.json() lanza JSONDecodeError (subclase de
    # ValueError), no RuntimeError. Sin este guard la excepcion escapa el
    # except del tool y crashea el FastMCP server.
    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"auth/token returned non-JSON response: {r.text[:200]}")
    # BC_NEW1 fix (adv-code Loop 2): JSON valido pero sin access_token (API
    # malformada, migracion, bug servidor) → KeyError escapa igual que JSONDecodeError.
    try:
        _jwt = data["access_token"]
    except KeyError:
        raise RuntimeError(f"auth/token response missing 'access_token' key: keys={list(data.keys())}")
    # El API devuelve `expires_in` en segundos; si no esta, asumimos 3600.
    expires_in = data.get("expires_in", 3600)
    _jwt_expires_at = now + float(expires_in)
    return _jwt


def _api_call(method: str, path: str, **kwargs) -> dict:
    """Wrapper HTTP con auth automatica y retry single-shot tras 401.

    method: GET | POST | PUT | DELETE
    path: relativo al API_URL (con leading slash)
    kwargs: params, json, etc. (pasados a httpx.Client.request)

    RF1 fix .HTTPError
    para que ConnectError, TimeoutException, RemoteProtocolError, etc. devuelvan
    RuntimeError limpio en lugar de propagar la excepcion de red al tool y crashear
    FastMCP.
    """
    try:
        with httpx.Client(timeout=ECODB_TIMEOUT) as client:
            token = _ensure_jwt(client)
            headers = kwargs.pop("headers", {}) or {}
            headers["Authorization"] = f"Bearer {token}"
            url = f"{ECODB_API_URL}{path}"
            r = client.request(method, url, headers=headers, **kwargs)
            if r.status_code == 401:
                # Refresh y reintentar UNA vez.
                token = _ensure_jwt(client, force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                r = client.request(method, url, headers=headers, **kwargs)
            if r.status_code >= 400:
                # Devolver detalle del API si lo hay; no exponemos internals
                # del MCP wrapper, solo lo que el API ya considera publicable.
                # OBS-1 fix (verificador): json.dumps(detail) en lugar de Python
                # repr para que el agente parsee facilmente el sub-JSON.
                try:
                    detail = r.json()
                    detail_str = json.dumps(detail, ensure_ascii=False)
                except Exception:
                    detail_str = r.text[:300]
                # UA4 fix (adv-code): http_status como atributo separado.
                err = RuntimeError(f"{method} {path} -> HTTP {r.status_code}: {detail_str}")
                err.http_status = r.status_code  # type: ignore[attr-defined]
                raise err
            if r.status_code == 204:
                return {"ok": True}
            return r.json()
    except httpx.HTTPError as e:
        # Errores de red (ConnectError, ReadTimeout, etc). Convertir a
        # RuntimeError generico para que las tools devuelvan _err limpio.
        raise RuntimeError(f"{method} {path}: network error ({type(e).__name__}): {e}") from e


def _normalize(obj):
    """Normaliza tipos no JSON-serializable para que FastMCP los maneje."""
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if hasattr(obj, 'isoformat'):  # datetime, date
        return obj.isoformat()
    if hasattr(obj, 'hex') and hasattr(obj, 'int'):  # UUID
        return str(obj)
    return obj


def _ok(data: Any) -> dict:
    """Respuesta para tools — dict normalizado directo a FastMCP."""
    return _normalize(data)


def _err(exc: Exception) -> dict:
    """Error — dict directo a FastMCP."""
    payload = {"error": str(exc)}
    status = getattr(exc, "http_status", None)
    if status is not None:
        payload["http_status"] = status
    return payload


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("ecodb", host=MCP_HOST, port=MCP_PORT)


@mcp.tool()
def save_memory(
    content: str,
    type: str,
    workspace_id: int = 1,
    project_id: int = 1,
    tags: Optional[list[str]] = None,
    visibility: str = "public",
    agent_identifier: Optional[str] = None,
    image_base64: Optional[str] = None,
    file_path: Optional[str] = None,
    image_path: Optional[str] = None,
) -> dict:
    """Guardar una memoria en EcoDB. Puede adjuntar un archivo de cualquier tipo.

    Args:
      content: texto de la memoria (max 16000 chars, obligatorio).
      type: OBLIGATORIO — uno de momento, decision, acuerdo, tecnico, descubrimiento, observacion, referencia.
      workspace_id: id del workspace (default 1 = system default).
      project_id: id del project (default 1 = general).
      tags: lista de etiquetas opcionales.
      visibility: public o private (default public).
      agent_identifier: opcional, nombre del agente que guarda.
      image_base64: opcional, imagen en base64 directa (PNG/JPEG/WebP).
      file_path: ruta a un archivo para adjuntar. Si es imagen raster
        (.png/.jpg/.jpeg/.webp/.gif/.bmp/.tiff) se embede para búsqueda visual
        Y se copia a media store. Si es otro tipo (.svg/.html/.pdf/.json/etc.)
        solo se copia a media store como asset accesible via media_path.
        Archivos de credenciales/config/ejecutables bloqueados por seguridad.
      image_path: alias de file_path (backward-compat). Si ambos presentes,
        file_path tiene prioridad.
    """
    payload = {
        "content": content,
        "type": type,
        "workspace_id": workspace_id,
        "project_id": project_id,
        "visibility": visibility,
        "tags": tags or [],
    }
    if agent_identifier is not None:
        payload["agent_identifier"] = agent_identifier
    resolved_file_path = file_path if file_path is not None else image_path
    real = None
    if resolved_file_path is not None and image_base64 is None:
        try:
            real = os.path.realpath(resolved_file_path)
            if not os.path.isfile(real):
                return _err(RuntimeError(f"file_path is not a regular file: {resolved_file_path}"))
            ext = Path(real).suffix.lower()
            if not ext or ext in _DENIED_EXTENSIONS:
                return _err(RuntimeError(f"file extension '{ext or '(none)'}' blocked for security"))
            fsize = os.path.getsize(real)
            if fsize > MAX_MEDIA_FILE_SIZE:
                return _err(RuntimeError(f"file exceeds max size ({fsize} > {MAX_MEDIA_FILE_SIZE} bytes)"))
            if ext in _EMBEDDABLE_IMAGE_EXTENSIONS:
                import base64 as _b64
                with open(real, "rb") as _f:
                    image_base64 = _b64.b64encode(_f.read()).decode("ascii")
        except (OSError, IOError) as _e:
            return _err(RuntimeError(f"cannot read file_path: {_e}"))
    if image_base64 is not None:
        payload["image_base64"] = image_base64
    try:
        data = _api_call("POST", "/memories", json=payload)
        # Media storage: copiar archivo a carpeta estable o guardar ruta original
        if resolved_file_path is not None and real is not None:
            file_ext = Path(real).suffix.lower() if real else ""
            if file_ext in _NO_COPY_EXTENSIONS:
                # Archivos con dependencias externas: guardar ruta original (no copiar)
                stored_path = real
            else:
                # Archivos autocontenidos: copiar a media store
                try:
                    stored_path = _copy_to_media_store(real, data["id"])
                except (RuntimeError, OSError, IOError) as _e:
                    data["media_storage_warning"] = f"memory saved but file copy failed: {_e}"
                    stored_path = None
            if stored_path:
                try:
                    _api_call("PUT", f"/memories/{quote(data['id'], safe='')}", json={"media_path": stored_path})
                    data["media_path"] = stored_path
                except RuntimeError as _e:
                    data["media_storage_warning"] = f"file copied to {stored_path} but DB update failed: {_e}"
        resp = {
            "status": "ok",
            "id": data["id"],
            "type": data.get("type"),
            "agent_identifier": data.get("agent_identifier"),
            "weight": data.get("weight"),
            "tags": data.get("tags", []),
            "created_at": data.get("created_at"),
        }
        if data.get("media_path"):
            resp["media_path"] = data["media_path"]
        return _ok(resp)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def search(
    query_text: Optional[str] = None,
    query_image: Optional[str] = None,
    query_type: Optional[str] = None,
    modality_filter: str = "all",
    limit: int = 20,
    max_images: int = 3,
    workspace_id: Optional[int] = None,
    project_id: Optional[int] = None,
    type: Optional[str] = None,
    user_id: Optional[int] = None,
    agent_identifier: Optional[str] = None,
    fecha_desde: Optional[str] = None,
    fecha_hasta: Optional[str] = None,
    expand_scope: bool = False,
    graph_discovery: bool = False,
    include_documents: bool = True,
    max_document_results: int = 3,
    tags: Optional[list[str]] = None,
    deep_factor: int = 2,
    cluster_mode: str = "none",
) -> list:
    """Busqueda semantica multimodal con motor GAMR (8 etapas).

    Motor de busqueda inteligente que combina similitud semantica, expansion
    por grafo de conocimiento, coherencia temporal y deteccion de
    contradicciones. Resultados ordenados por score compuesto GAMR.

    Args:
      query_text: texto de busqueda (>=3, <=2000 chars). Uno de query_text o
        query_image es obligatorio.
      query_image: imagen en base64 (PNG/JPEG/WebP) para busqueda visual.
        Cross-modal: texto encuentra imagenes y viceversa.
      query_type: tipo de consulta — afecta pesos del score compuesto:
        - "factual": prioriza frescura (0.30) y semantica (0.35). Para
          preguntas de hechos actuales: "que es X", "donde esta Y".
        - "historical": no penaliza antiguedad (freshness 0.10), prioriza
          peso de memoria (0.30). Para linea temporal: "cuando paso X".
        - "analytical": maximiza grafo (0.30) y semantica (0.30). Para
          analisis: "por que X", "diferencia entre X e Y".
        - "contextual": equilibrado. Default si no se especifica.
        Si no se proporciona, se clasifica automaticamente por heuristicas.
      modality_filter: filtrar por modalidad. "all" (default), "text",
        "image", "audio".
      limit: max resultados (1-100, default 20).
      max_images: max imagenes inline (default 3). 0 = sin imagenes.
      workspace_id: filtrar a un workspace.
      project_id: filtrar a un project.
      type: filtrar por tipo de memoria (momento/decision/acuerdo/tecnico/
        descubrimiento/observacion/referencia).
      user_id: filtrar por creador.
      agent_identifier: filter by agent identifier.
      fecha_desde: ISO 8601, memorias creadas desde esta fecha.
      fecha_hasta: ISO 8601, memorias creadas hasta esta fecha.
      expand_scope: override visibilidad por jerarquia estricta.
      graph_discovery: si true, GAMR añade memorias descubiertas via grafo
        de conocimiento que no aparecieron en busqueda semantica. Util cuando
        el grafo es denso. Las memorias descubiertas tienen semantic_score=0.0
        y matched_modality="graph". Default false.
      tags: filtrar por tags (AND logico — memoria debe tener TODOS). Ej: ["ancla-visual", "landing"].
      cluster_mode: enriquecer con clusters de memoria. "none" (default, sin
        cambios), "include" (related_clusters con narrative_preview), "mixed"
        (merged_results con memorias y clusters intercalados por score).

    Respuesta incluye por cada resultado:
      - score: score compuesto GAMR (lo que ordena los resultados)
      - semantic_score: similitud coseno pura
      - graph_score: proximidad en grafo de conocimiento
      - freshness_score: frescura temporal (1.0 = reciente, 0.0 = >1 año)
      - score_breakdown: desglose del score por componente

    Respuesta incluye metadata:
      - query_type: tipo clasificado (auto o manual)
      - graph_context: entidades del grafo activadas
      - contradictions: pares de memorias potencialmente contradictorias
        (similitud >0.85, >1 dia de diferencia, mismo tipo)
    """
    from mcp.types import TextContent, ImageContent
    import base64 as _b64
    from pathlib import Path as _Path

    if query_text is None and query_image is None:
        return [TextContent(type="text", text=json.dumps({"error": "at least one of query_text or query_image is required"}, ensure_ascii=False))]
    if limit < 1 or limit > 100:
        return [TextContent(type="text", text=json.dumps({"error": "limit must be between 1 and 100"}, ensure_ascii=False))]
    if deep_factor < 1 or deep_factor > 10:
        return [TextContent(type="text", text=json.dumps({"error": "deep_factor must be between 1 and 10"}, ensure_ascii=False))]
    if cluster_mode not in ("none", "include", "mixed"):
        return [TextContent(type="text", text=json.dumps({"error": "cluster_mode must be none|include|mixed"}, ensure_ascii=False))]
    payload: dict = {"limit": limit, "expand_scope": expand_scope, "modality_filter": modality_filter}
    if query_text is not None:
        payload["query_text"] = query_text
    if query_image is not None:
        payload["query_image"] = query_image
    if query_type is not None:
        payload["query_type"] = query_type
    if workspace_id is not None:
        payload["workspace_id"] = workspace_id
    if project_id is not None:
        payload["project_id"] = project_id
    if type is not None:
        payload["type"] = type
    if user_id is not None:
        payload["user_id"] = user_id
    if agent_identifier is not None:
        payload["agent_identifier"] = agent_identifier
    if fecha_desde is not None:
        payload["fecha_desde"] = fecha_desde
    if fecha_hasta is not None:
        payload["fecha_hasta"] = fecha_hasta
    if graph_discovery:
        payload["graph_discovery"] = True
    if include_documents:
        payload["include_documents"] = True
        payload["max_document_results"] = max_document_results
    if tags:
        payload["tags"] = tags
    if deep_factor != 2:
        payload["deep_factor"] = deep_factor
    if cluster_mode != "none":
        payload["cluster_mode"] = cluster_mode
    try:
        data = _api_call("POST", "/search", json=payload)
    except RuntimeError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]

    results = data.get("results", [])
    for r in results:
        if r.get("source_type") == "document_chunk":
            safe_content = r.get("content", "").replace("</doc>", "&lt;/doc&gt;")
            r["content"] = f'<doc source="{r.get("id")}">{safe_content}</doc>'

    doc_counts: dict = {}
    for r in results[:10]:
        if r.get("source_type") == "document_chunk" and r.get("score", 0) > 0.5:
            doc_id = r.get("document_id") or r.get("metadata", {}).get("document_id")
            if doc_id:
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    recommendations = [
        f"Document {doc_id} appears {cnt} times in top results — consider reading it fully with read_document or search_in_document"
        for doc_id, cnt in doc_counts.items()
        if cnt >= 2
    ]
    if recommendations:
        results.insert(0, {"_recommendation": "; ".join(recommendations)})

    content_blocks = [TextContent(type="text", text=json.dumps(_normalize(data), ensure_ascii=False, indent=2))]

    images_added = 0
    asset_paths = []
    for result in data.get("results", []):
        media_path = result.get("media_path")
        if not media_path:
            continue
        p = _Path(os.path.realpath(media_path))
        if not p.exists():
            continue
        in_store = p.is_relative_to(MEDIA_STORE_DIR)
        suffix = p.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp",
                ".bmp": "image/bmp", ".tiff": "image/tiff"}.get(suffix)
        if mime and in_store and images_added < max_images:
            try:
                img_data = _b64.b64encode(p.read_bytes()).decode("ascii")
                content_blocks.append(ImageContent(
                    type="image", data=img_data, mimeType=mime,
                ))
                images_added += 1
            except (OSError, IOError):
                continue
        else:
            asset_paths.append(media_path)

    if asset_paths:
        content_blocks.append(TextContent(type="text", text=(
            f"\n📎 {len(asset_paths)} documento(s) adjunto(s) en media_path. "
            "Los recuerdos resumen información pero no sustituyen los documentos originales. "
            "Si un resultado relevante apunta a un archivo, consulta el documento completo con Read "
            "— especialmente si varios recuerdos convergen en el mismo tema. "
            "Usa criterio: resultados con score alto sobre tu tema de trabajo merecen consulta; "
            "resultados con score bajo o tangenciales no. El recuerdo es el mapa, el documento es el territorio."
        )))

    return content_blocks


@mcp.tool()
def search_recent(
    limit: int = 20,
    max_images: int = 3,
    workspace_id: Optional[int] = None,
    project_id: Optional[int] = None,
    user_id: Optional[int] = None,
    agent_identifier: Optional[str] = None,
    fecha_desde: Optional[str] = None,
    fecha_hasta: Optional[str] = None,
    tags: Optional[list[str]] = None,
    expand_scope: bool = False,
) -> list:
    """Listar memorias recientes filtradas por permisos del actor.

    Args:
      limit: max resultados (1-100, default 20).
      max_images: max imagenes devueltas inline (default 3). 0 = sin imagenes inline.
      workspace_id: opcional, filtrar a un workspace concreto.
      project_id: opcional, filtrar a un project concreto.
      user_id: opcional, filtrar por creador (.
      agent_identifier: opcional, filtrar por agente (.
      fecha_desde: opcional ISO 8601 (.
      fecha_hasta: opcional ISO 8601 (.
      tags: opcional, lista de tags — AND lógico (memoria debe tener TODOS). Ejemplo: ["landing", "status:approved"].
      expand_scope: opcional, override visibility por jerarquía estricta. Audit
        log obligatorio. Default false.
    """
    from mcp.types import TextContent, ImageContent
    import base64 as _b64
    from pathlib import Path as _Path

    params: dict = {"limit": limit, "expand_scope": expand_scope}
    if workspace_id is not None:
        params["workspace_id"] = workspace_id
    if project_id is not None:
        params["project_id"] = project_id
    if user_id is not None:
        params["user_id"] = user_id
    if agent_identifier is not None:
        params["agent_identifier"] = agent_identifier
    if fecha_desde is not None:
        params["fecha_desde"] = fecha_desde
    if fecha_hasta is not None:
        params["fecha_hasta"] = fecha_hasta
    if tags is not None:
        params["tags"] = tags
    try:
        data = _api_call("GET", "/memories/recent", params=params)
    except RuntimeError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]

    content_blocks = [TextContent(type="text", text=json.dumps(_normalize(data), ensure_ascii=False, indent=2))]

    images_added = 0
    asset_paths = []
    for item in data.get("items", []):
        media_path = item.get("media_path")
        if not media_path:
            continue
        p = _Path(os.path.realpath(media_path))
        if not p.exists():
            continue
        in_store = p.is_relative_to(MEDIA_STORE_DIR)
        suffix = p.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp",
                ".bmp": "image/bmp", ".tiff": "image/tiff"}.get(suffix)
        if mime and in_store and images_added < max_images:
            try:
                img_data = _b64.b64encode(p.read_bytes()).decode("ascii")
                content_blocks.append(ImageContent(
                    type="image", data=img_data, mimeType=mime,
                ))
                images_added += 1
            except (OSError, IOError):
                continue
        else:
            asset_paths.append(media_path)

    if asset_paths:
        content_blocks.append(TextContent(type="text", text=(
            f"\n📎 {len(asset_paths)} documento(s) adjunto(s) en media_path. "
            "Los recuerdos resumen información pero no sustituyen los documentos originales. "
            "Si un resultado relevante apunta a un archivo, consulta el documento completo con Read "
            "— especialmente si varios recuerdos convergen en el mismo tema. "
            "Usa criterio: resultados con score alto sobre tu tema de trabajo merecen consulta; "
            "resultados con score bajo o tangenciales no. El recuerdo es el mapa, el documento es el territorio."
        )))

    return content_blocks


@mcp.tool()
def get_relevant_context(
    prompt_text: str,
    max_results: int = 3,
    threshold: float = 0.6,
) -> dict:
    """Obtiene contexto relevante de EcoDB para inyección automática.

    Busca memorias/documentos relevantes al prompt dado. Diseñado para
    ser llamado automáticamente al inicio de cada turno de trabajo.
    Solo devuelve resultados si score > threshold.

    Args:
      prompt_text: texto del prompt/contexto actual del agente.
      max_results: máximo resultados (1-5, default 3).
      threshold: score mínimo para incluir (0.0-1.0, default 0.6).
    """
    import uuid as _uuid

    enable = os.environ.get("ENABLE_CONTEXT_INJECTION", "false").lower() in ("true", "1", "yes", "on")
    if not enable:
        return _ok({"injected": False, "reason": "context injection disabled"})

    try:
        results_raw = _api_call("POST", "/search", json={
            "query_text": prompt_text[:500],
            "limit": max_results * 2,
            "include_documents": True,
        })
    except RuntimeError as e:
        return _err(e)

    filtered = [r for r in results_raw.get("results", []) if r.get("score", 0) >= threshold][:max_results]

    if not filtered:
        return _ok({"injected": False, "reason": "no results above threshold"})

    def _sanitize_xml(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    injection_id = _uuid.uuid4().hex[:8]
    lines = []
    memory_ids = []
    scores = []
    for r in filtered:
        rid = str(r.get("id", ""))
        memory_ids.append(rid)
        scores.append(r.get("score", 0))
        rtype = _sanitize_xml(r.get("type", "?"))
        content_preview = _sanitize_xml(r.get("content", "")[:150])
        source = _sanitize_xml(r.get("source_type", "memory"))
        lines.append(f"[EcoDB:{injection_id}] {source}/{rtype}: {content_preview}")

    context_block = (
        "<ecodb_context note=\"puede ser relevante o no — ignóralo si no ayuda\">\n"
        + "\n".join(f"• {l}" for l in lines)
        + "\n</ecodb_context>"
    )

    import hashlib as _hashlib
    prompt_hash = _hashlib.md5(prompt_text.encode()).hexdigest()[:12]

    # Fire-and-forget telemetry (best-effort)
    try:
        _api_call("POST", "/telemetry/record", json={
            "injection_id": injection_id,
            "memory_ids": memory_ids,
            "scores": scores,
            "prompt_hash": prompt_hash,
        })
    except Exception as _e:
        import logging as _log
        _log.getLogger("ecodb.mcp").warning("telemetry record failed: %r", _e)

    return _ok({
        "injected": True,
        "injection_id": injection_id,
        "count": len(filtered),
        "context_block": context_block,
    })


@mcp.tool()
def read_memory(memory_id: str, expand_scope: bool = False) -> list:
    """Leer una memoria por ID. Incrementa access_count.

    Args:
      memory_id: UUID de la memoria.
      expand_scope: opcional, override visibility por jerarquía estricta (Tarea
        2.10). Si la memoria es private y el actor está jerárquicamente sobre
        el creador en el árbol del workspace/project, expand_scope=true permite
        leerla. Audit log obligatorio. Default false (comportamiento estricto).
    """
    from mcp.types import TextContent, ImageContent
    import base64 as _b64
    from pathlib import Path as _Path

    # VS2-MCP fix .
    params = {"expand_scope": expand_scope} if expand_scope else None
    try:
        data = _api_call("GET", f"/memories/{quote(memory_id, safe='')}", params=params)
    except RuntimeError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]

    content_blocks = [TextContent(type="text", text=json.dumps(_normalize(data), ensure_ascii=False, indent=2))]

    media_path = data.get("media_path")
    if media_path:
        p = _Path(os.path.realpath(media_path))
        in_store = p.is_relative_to(MEDIA_STORE_DIR)
        if p.exists():
            suffix = p.suffix.lower()
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".gif": "image/gif", ".webp": "image/webp",
                    ".bmp": "image/bmp", ".tiff": "image/tiff"}.get(suffix)
            if mime and in_store:
                try:
                    img_data = _b64.b64encode(p.read_bytes()).decode("ascii")
                    content_blocks.append(ImageContent(
                        type="image", data=img_data, mimeType=mime,
                    ))
                except (OSError, IOError):
                    pass
            else:
                content_blocks.append(TextContent(type="text", text=(
                    f"\n📎 Documento adjunto: {media_path}\n"
                    "Este recuerdo apunta a un documento original. Si es relevante para tu trabajo, "
                    "consulta el documento completo con Read. "
                    "El recuerdo es el mapa, el documento es el territorio."
                )))

    return content_blocks


@mcp.tool()
def save_triple(
    subject: str,
    predicate: str,
    object: str,
    author: Optional[str] = None,
) -> dict:
    """Guardar una tripleta sujeto-predicado-objeto en el grafo (SQL + AGE atomico).

    Args:
      subject: nombre del nodo sujeto (max 500 chars).
      predicate: nombre de la relacion (max 200 chars).
      object: nombre del nodo objeto (max 500 chars).
      author: opcional, autor de la tripleta.
    """
    # Pipeline de normalización de predicados (Fase 3b, 5 etapas)
    if not _alias_cache_loaded:
        _load_alias_cache()
    resolution = _resolve_predicate(predicate)
    canonical_predicate = resolution["canonical"]

    payload = {"subject": subject, "predicate": canonical_predicate, "object": object}
    if author is not None:
        payload["author"] = author
    if resolution["needs_review"]:
        payload["needs_review"] = True
        payload["mapper_confidence"] = resolution["confidence"]
        payload["original_predicate"] = resolution["original"]
    elif resolution["original"] != canonical_predicate:
        payload["mapper_confidence"] = resolution["confidence"]
        payload["original_predicate"] = resolution["original"]
    try:
        data = _api_call("POST", "/graph/triples", json=payload)
        resp = {
            "status": "ok",
            "id": data.get("id"),
            "subject": data.get("subject") or data.get("subject_name"),
            "predicate": canonical_predicate,
            "object": data.get("object") or data.get("object_name"),
        }
        if resolution["original"] != canonical_predicate:
            resp["original_predicate"] = resolution["original"]
            resp["canonical_predicate"] = canonical_predicate
            resp["mapper_confidence"] = resolution["confidence"]
        if resolution["needs_review"]:
            resp["needs_review"] = True
        return _ok(resp)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def neighbors(node: str, depth: int = 1) -> dict:
    """Vecinos de un nodo en el grafo a 1-N saltos (default 1).

    Args:
      node: nombre del nodo central.
      depth: 1-3 (default 1). Mayor profundidad = mas resultados pero mas lento.
    """
    # UA2 fix .
    # El API tiene su propia validacion, pero un agente con depth=50 dispara un
    # roundtrip caro innecesario. Mejor reject barato aqui.
    depth = max(1, min(int(depth), 3))
    # VS2-MCP fix (adv-seg): quote previene path traversal en node — un nodo
    # llamado "../../admin" se trata como string literal en la URL, no como
    # segmento de path. Permite tambien nombres con caracteres especiales.
    try:
        data = _api_call("GET", f"/graph/neighbors/{quote(node, safe='')}", params={"depth": depth})
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


# ---------------------------------------------------------------------------
# — Sprint Paridad parcial Opcion A (identidad + grafo navegacion)
# ---------------------------------------------------------------------------

@mcp.tool()
def load_identity(agent_identifier: str, version: Optional[int] = None) -> dict:
    """Cargar identidad de un agente desde EcoDB.

    Devuelve los fragmentos de identidad concatenados como un único string,
    separados por "\\n\\n---\\n\\n" entre fragmentos. Orden narrativo (ascendente
    por fragment_idx) — la identidad tiene un orden y se pierde si llegan
    desordenados (requirement vinculante de Eco).

    Args:
      agent_identifier: Agent identifier.
      version: optional. If None, returns the maximum version (current). If
        an INT >=1 is passed, returns that specific historical version.

    Permissions: only the agent's owner (agents.user_id == jwt.sub) or super.
    Others → 404 (anti-discovery).

    If the agent has no fragments saved yet, returns an empty string.
    """
    path = f"/agents/{quote(agent_identifier, safe='')}/identity"
    params = {"version": version} if version is not None else None
    try:
        data = _api_call("GET", path, params=params)
    except RuntimeError as e:
        return _err(e)
    fragments = data.get("fragments", []) or []
    # Concatenate fragments as plain text with separator — no synthetic headers
    # (the schema has no title/hash fields; inventing them would be false metadata).
    text = "\n\n---\n\n".join(f.get("content", "") for f in fragments)
    return _ok({
        "agent_identifier": data.get("agent_identifier", agent_identifier),
        "agent_id": data.get("agent_id"),
        "version": data.get("version", 0),
        "text": text,
        "fragments_count": len(fragments),
    })


@mcp.tool()
def save_identity(agent_identifier: str, fragments: list[str]) -> dict:
    """Guardar nueva version de identidad (snapshot completo) para un agente.

    Args:
      agent_identifier: Agent identifier.
      fragments: list of strings with identity fragments in narrative order.
        narrativo. fragment_idx se asigna automaticamente segun la posicion
        en la lista (0, 1, 2, ...). El cliente NO gestiona idx.

    La nueva version se auto-incrementa (max actual + 1). El snapshot es
    completo — no se parchea por fragmento individual. Para rollback, cargar
    una version anterior.

    Permisos: solo el user dueño del agent (agents.user_id == jwt.sub) o super.
    """
    path = f"/agents/{quote(agent_identifier, safe='')}/identity"
    payload = {"fragments": fragments}
    try:
        data = _api_call("POST", path, json=payload)
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def path_between(source: str, target: str, max_depth: int = 6) -> dict:
    """Camino mas corto entre dos nodos del grafo (BFS via Cypher).

    Args:
      source: nombre del nodo origen.
      target: nombre del nodo destino.
      max_depth: profundidad maxima 1-10 (default 6).

    Devuelve el camino como secuencia de nodos y aristas, o vacio si no hay
    conexion en max_depth saltos.
    """
    max_depth = max(1, min(int(max_depth), 10))
    params = {"source": source, "target": target, "max_depth": max_depth}
    try:
        data = _api_call("GET", "/graph/path", params=params)
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def search_nodes(query: str, limit: int = 20) -> dict:
    """Fuzzy search de nodos del grafo por nombre (pg_trgm).

    Args:
      query: substring o nombre aproximado (min 3 caracteres — pg_trgm GIN index
        requiere >=3 chars para extraer trigramas eficientemente).
      limit: max resultados 1-100 (default 20).

    Util antes de pedir neighbors o caminos cuando no recuerdas el nombre exacto
    del nodo. Devuelve ranking por similitud trigrama.
    """
    if len(query.strip()) < 3:
        return _err(RuntimeError("query must be at least 3 characters"))
    limit = max(1, min(int(limit), 100))
    params = {"q": query, "limit": limit}
    try:
        data = _api_call("GET", "/graph/search", params=params)
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def view_image(memory_id: str) -> Image:
    """Ver la imagen asociada a una memoria. Devuelve la imagen REAL que Claude
    puede ver directamente — no texto, no base64, la imagen renderizada.

    Args:
      memory_id: UUID de la memoria que tiene media_path con imagen.

    Si la memoria no tiene media_path o el archivo no existe, devuelve error.
    Usa tras search() cuando un resultado tiene matched_modality='image' o
    media_path no null.
    """
    try:
        data = _api_call("GET", f"/memories/{quote(memory_id, safe='')}")
    except RuntimeError as e:
        raise ValueError(str(e))
    media_path = data.get("media_path")
    if not media_path:
        raise ValueError(f"memory {memory_id} has no media_path")
    p = Path(os.path.realpath(media_path))
    if not p.is_relative_to(MEDIA_STORE_DIR):
        raise ValueError(f"media_path outside of media store")
    if not p.exists():
        raise ValueError(f"image file not found: {media_path}")
    return Image(path=str(p))


# ---------------------------------------------------------------------------
# — 5 wrappers MCP paridad (tools sobre endpoints existentes)
# ---------------------------------------------------------------------------

@mcp.tool()
def save_triples_batch(
    triples: list[dict],
) -> dict:
    """Guardar multiples tripletas en una sola transaccion atomica (batch).

    Args:
      triples: lista de dicts, cada uno con keys "subject", "predicate", "object"
        (obligatorios) y "author" (opcional). Max 100 por batch.
    """
    if len(triples) > 100:
        return _err(RuntimeError("batch size exceeds max 100 triples"))
    if not _alias_cache_loaded:
        _load_alias_cache()
    resolved_triples = []
    for i, t in enumerate(triples):
        subj = t.get("subject", "")
        pred = t.get("predicate", "")
        obj = t.get("object", "")
        if not subj or not pred or not obj:
            return _err(RuntimeError(f"triple[{i}]: subject, predicate, object are required and cannot be empty"))
        res = _resolve_predicate(pred)
        resolved = dict(t)
        resolved["predicate"] = res["canonical"]
        if res["needs_review"]:
            resolved["needs_review"] = True
            resolved["mapper_confidence"] = res["confidence"]
            resolved["original_predicate"] = res["original"]
        elif res["original"] != res["canonical"]:
            resolved["mapper_confidence"] = res["confidence"]
            resolved["original_predicate"] = res["original"]
        resolved_triples.append(resolved)
    try:
        data = _api_call("POST", "/graph/triples/batch", json={"triples": resolved_triples})
        return _ok({
            "status": "ok",
            "count": data.get("created", 0),
        })
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def delete_memory(memory_id: str) -> dict:
    """Soft-delete de una memoria (va a papelera). Requiere permisos de escritura.

    Args:
      memory_id: UUID de la memoria a borrar.
    """
    try:
        data = _api_call("DELETE", f"/memories/{quote(memory_id, safe='')}")
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def unarchive_memory(memory_id: str) -> dict:
    """Desarchivar una memoria archivada. Transición archived→active.

    Args:
      memory_id: UUID de la memoria a desarchivar.
    """
    try:
        data = _api_call("PUT", f"/memories/{quote(memory_id, safe='')}/unarchive")
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def delete_triple(triple_id: int) -> dict:
    """Elimina una tripleta del grafo (SQL + AGE atomico).

    Args:
      triple_id: ID numerico de la tripleta.
    """
    try:
        data = _api_call("DELETE", f"/graph/triples/{int(triple_id)}")
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def graph_status() -> dict:
    """Estadisticas del grafo: numero de nodos, tripletas y predicados unicos.
    Util como GPS para saber el tamaño y forma del grafo actual.
    """
    try:
        data = _api_call("GET", "/graph/stats")
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Task 4.11 — Document ingestion tools
# ---------------------------------------------------------------------------

_DOC_EXT_MAP: dict[str, str] = {
    ".pdf": "pdf", ".docx": "docx", ".doc": "docx",
    ".html": "html", ".htm": "html",
    ".md": "md", ".markdown": "markdown",
    ".txt": "txt", ".text": "txt",
    ".mp3": "audio", ".wav": "audio", ".m4a": "audio",
    ".ogg": "audio", ".flac": "audio", ".mp4": "audio",
}


@mcp.tool()
def register_document(
    uri: str,
    project_id: int,
    doc_type: Optional[str] = None,
    visibility: str = "public",
) -> dict:
    """Registra un documento en EcoDB y lo encola para indexacion.

    Copia el archivo al media store permanente, lo registra en la BD con
    status='queued', y notifica al worker para procesar el pipeline de
    ingestion (parse → chunk → GLiNER → embed → graph).

    Args:
      uri: ruta local al archivo (PDF, DOCX, HTML, MD, TXT, audio).
      project_id: proyecto al que pertenece el documento.
      doc_type: tipo de documento. Si None, se infiere de la extension.
      visibility: "public" (default) o "private".

    Returns: {status, document_id, uri, doc_type}
    """
    from pathlib import Path as _Path
    import uuid as _uuid_mod

    # B4-3: reject URL schemes + traversal + denied extensions
    if _URL_SCHEME_RE.match(uri):
        return _err(RuntimeError(f"URL schemes not allowed in uri: {uri[:50]}"))
    real = os.path.realpath(uri)
    if not os.path.isfile(real):
        return _err(RuntimeError(f"file not found: {uri}"))
    suffix = _Path(real).suffix.lower()
    if not suffix or suffix in _DENIED_EXTENSIONS:
        return _err(RuntimeError(f"file extension '{suffix or '(none)'}' blocked for security"))

    if doc_type is None:
        doc_type = _DOC_EXT_MAP.get(suffix, "txt")

    stored_path: str | None = None
    try:
        temp_id = f"doc_{_uuid_mod.uuid4()}"
        stored_path = _copy_to_media_store(real, temp_id)
    except (RuntimeError, OSError, IOError) as e:
        return _err(RuntimeError(f"file copy failed: {e}"))

    filename = _Path(real).name
    try:
        data = _api_call("POST", "/documents", json={
            "uri": stored_path,
            "filename": filename,
            "doc_type": doc_type,
            "project_id": project_id,
            "visibility": visibility,
        })
    except RuntimeError as e:
        # BC1: cleanup media store copy on API failure
        if stored_path:
            try:
                os.unlink(stored_path)
            except OSError:
                pass
        return _err(e)
    return _ok({"status": data.get("status"), "document_id": str(data.get("id")),
                "uri": stored_path, "doc_type": doc_type})


@mcp.tool()
def document_status(document_id: str) -> dict:
    """Estado de procesamiento de un documento: status, metricas, última indexacion.

    Args:
      document_id: UUID del documento.
    """
    if not _UUID_RE.match(document_id):
        return _err(RuntimeError("document_id must be a valid UUID"))
    try:
        data = _api_call("GET", f"/documents/{quote(str(document_id), safe='')}")
        return _ok({
            "document_id": str(data.get("id")),
            "status": data.get("status"),
            "doc_type": data.get("doc_type"),
            "filename": data.get("filename"),
            "retry_count": data.get("retry_count"),
            "last_indexed": data.get("last_indexed"),
            "processing_metrics": data.get("processing_metrics"),
        })
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def list_documents(
    project_id: Optional[int] = None,
    workspace_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Lista documentos accesibles, con filtros opcionales.

    Args:
      project_id: filtrar por proyecto.
      workspace_id: filtrar por workspace.
      status: filtrar por estado (queued/processing/indexed/failed).
      limit: max resultados (1-100, default 20).
      offset: saltar N primeros resultados (default 0).
    """
    limit = min(max(1, limit), 100)
    params: dict = {"limit": limit, "offset": max(0, offset)}
    if project_id is not None:
        params["project_id"] = project_id
    if workspace_id is not None:
        params["workspace_id"] = workspace_id
    if status is not None:
        params["status"] = status
    try:
        data = _api_call("GET", "/documents", params=params)
        return _ok({"documents": _normalize(data), "count": len(data) if isinstance(data, list) else 0})
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def search_in_document(
    document_id: str,
    query_text: str,
    limit: int = 5,
) -> dict:
    """Busqueda semantica dentro de un documento especifico.

    Devuelve los chunks del documento mas relevantes para la consulta,
    ordenados por similitud coseno.

    Args:
      document_id: UUID del documento.
      query_text: texto de busqueda.
      limit: max chunks (1-50, default 5).
    """
    if not _UUID_RE.match(document_id):
        return _err(RuntimeError("document_id must be a valid UUID"))
    limit = min(max(1, limit), 50)
    try:
        data = _api_call("POST", f"/search/document/{quote(str(document_id), safe='')}",
                         json={"query_text": query_text, "limit": limit})
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def read_document(
    document_id: str,
    start_chunk: int = 0,
    limit: int = 50,
) -> dict:
    """Lee el contenido de un documento concatenando sus chunks en orden.

    Util para leer documentos indexados sin acceder al archivo original.
    Si el documento es largo, usa start_chunk + limit para paginar.

    Args:
      document_id: UUID del documento.
      start_chunk: chunk desde el que empezar (default 0).
      limit: max chunks a leer (1-200, default 50).

    Returns: {content, chunks_returned, total_chunks, truncated}
    """
    if not _UUID_RE.match(document_id):
        return _err(RuntimeError("document_id must be a valid UUID"))
    limit = min(max(1, limit), 200)
    try:
        data = _api_call("GET", f"/documents/{quote(str(document_id), safe='')}/chunks",
                         params={"start": start_chunk, "limit": limit})
        chunks = data.get("chunks", [])
        content = "\n\n".join(c.get("content", "") for c in chunks)
        return _ok({
            "content": content,
            "chunks_returned": data.get("chunks_returned", len(chunks)),
            "total_chunks": data.get("total_chunks", 0),
            "truncated": data.get("truncated", False),
        })
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def validate_link(memory_id: str, document_id: str) -> dict:
    """Validar un auto-link entre memoria y documento.

    Confirma que el vínculo automático es correcto. Cambia validated=true,
    lo que sube el source_score de 0.5x a 1.0x en GAMR Etapa 5.

    Args:
      memory_id: UUID de la memoria.
      document_id: UUID del documento.
    """
    if not _UUID_RE.match(memory_id) or not _UUID_RE.match(document_id):
        return _err(RuntimeError("memory_id and document_id must be valid UUIDs"))
    try:
        resp = _api_call("PUT", f"/memories/{quote(str(memory_id), safe='')}/links/{quote(str(document_id), safe='')}/validate")
        return _ok(resp)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def review_alias_candidate(candidate_id: int, decision: str, merge: bool = False, reason: str = "") -> dict:
    """Revisar candidato de alias de entidad. Super-only.

    Aprueba o rechaza un candidato detectado por GLiNER.
    Si decision='approved' y merge=True, fusiona la entidad al nodo canonical.

    Args:
      candidate_id: ID del alias candidate (ver alias-candidates).
      decision: 'approved' o 'rejected'.
      merge: Si True y approved, ejecutar merge inmediato.
      reason: Motivo opcional (max 500 chars).
    """
    body: dict = {"status": decision, "merge": merge}
    if reason:
        body["reason"] = reason
    try:
        resp = _api_call("PUT", f"/admin/alias-candidates/{candidate_id}", json=body)
        return _ok(resp)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def merge_entities(source_node_id: int, target_node_id: int, reason: str = "") -> dict:
    """Fusionar entidad fuente en entidad destino (soft merge). Super-only.

    Marca source_node_id como merged, apuntando a target (canonico).
    Compresión de cadena: si target ya fue mergeado, se resuelve al canonical.

    Args:
      source_node_id: ID SQL del nodo a marcar como merged.
      target_node_id: ID SQL del nodo canonical (destino).
      reason: Motivo opcional (max 500 chars).
    """
    body: dict = {"source_node_id": source_node_id, "target_node_id": target_node_id}
    if reason:
        body["reason"] = reason
    try:
        resp = _api_call("POST", "/admin/merge-entities", json=body)
        return _ok(resp)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def undo_merge(source_node_id: int) -> dict:
    """Deshacer el último merge activo de un nodo. Super-only.

    Restaura source_node_id a status='active' y limpia merged_into.
    Solo funciona si hay un merge activo (undone_at IS NULL).

    Args:
      source_node_id: ID SQL del nodo a restaurar.
    """
    try:
        resp = _api_call("POST", "/admin/undo-merge", json={"source_node_id": source_node_id})
        return _ok(resp)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def classify_document(document_id: str, trust_tier: int) -> dict:
    """Establecer nivel de confianza de un documento. Super-only.

    trust_tier: 0=no confiable, 1=default, 2=verificado, 3=gold.
    Afecta al weight efectivo y decay en GAMR Etapa 5 cuando ENABLE_TRUST_TIERS=true.

    Args:
      document_id: UUID del documento.
      trust_tier: Nivel 0-3.
    """
    if not _UUID_RE.match(document_id):
        return _err(RuntimeError("document_id must be a valid UUID"))
    if trust_tier not in (0, 1, 2, 3):
        return _err(ValueError("trust_tier must be 0-3"))
    try:
        resp = _api_call("PUT", f"/admin/documents/{quote(str(document_id), safe='')}/trust-tier",
                         json={"trust_tier": trust_tier})
        return _ok(resp)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def confirm_document_relation(source_id: str, target_id: str) -> dict:
    """Confirmar relación detectada entre dos documentos. Super-only.

    Marca confirmed_by con el usuario actual. Las relaciones no confirmadas
    se purgan tras 90 días en el ciclo de gobernanza.

    Args:
      source_id: UUID del documento fuente.
      target_id: UUID del documento relacionado.
    """
    try:
        resp = _api_call("PUT", "/admin/related-documents/confirm",
                         json={"source_id": source_id, "target_id": target_id})
        return _ok(resp)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def reindex_document(document_id: str) -> dict:
    """Reencola un documento para que el worker lo procese de nuevo.

    Util para forzar re-indexacion tras un fallo o cambio de configuracion.
    Resetea retry_count y status a 'queued'.

    Args:
      document_id: UUID del documento.
    """
    if not _UUID_RE.match(document_id):
        return _err(RuntimeError("document_id must be a valid UUID"))
    try:
        data = _api_call("PUT", f"/documents/{quote(str(document_id), safe='')}/reindex")
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def unlink_document(document_id: str) -> dict:
    """Elimina (soft delete) un documento de EcoDB.

    El documento queda con status='deleted' y deja de aparecer en busquedas.
    Los chunks y entity_links se conservan en BD pero son ignorados.

    Args:
      document_id: UUID del documento.
    """
    if not _UUID_RE.match(document_id):
        return _err(RuntimeError("document_id must be a valid UUID"))
    try:
        _api_call("DELETE", f"/documents/{quote(str(document_id), safe='')}")
        return _ok({"status": "deleted", "document_id": document_id})
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def seed_dictionary(
    name: str,
    entity_type: str,
    notes: Optional[str] = None,
) -> dict:
    """Añadir entidad al diccionario del grafo.

    Usar cuando se descubre una entidad nueva durante extracción de
    tripletas o al cerrar sesión con temas nuevos. Requiere aprobación
    humana del tipo antes de llamar.

    Args:
      name: nombre de la entidad (tal como aparece en texto).
      entity_type: uno de persona, agente_ia, organizacion, lugar,
        producto, proyecto, tecnologia, concepto, evento, artefacto,
        modelo_ia, metodologia.
      notes: nota opcional sobre la entidad.
    """
    try:
        data = _api_call("POST", "/admin/entity-dictionary", json={
            "name": name,
            "entity_type": entity_type,
            "notes": notes,
        })
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def get_graph_vocabulary() -> dict:
    """Vocabulario del grafo: entidades del diccionario + predicados aprobados.

    Devuelve las entidades aceptadas (name + type) y los predicados
    autorizados (name + description). Usar antes de generar cualquier
    tripleta para respetar el vocabulario gobernado.
    """
    try:
        data = _api_call("GET", "/admin/graph-vocabulary")
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Metacognition cluster tools (Memory Agent v1.3)
# ---------------------------------------------------------------------------

@mcp.tool()
def search_clusters(query_text: str, agent_identifier: Optional[str] = None,
                    level: Optional[str] = None, limit: int = 10) -> dict:
    """Buscar clusters por similitud semántica (coseno sobre centroides + BM25 en labels).

    Args:
      query_text: texto de búsqueda (3-2000 chars).
      agent_identifier: filtrar por agente. Sin él, solo clusters SIN_AUTOR
        (genéricos/técnicos) para no contaminar con memoria de otro agente.
      level: weekly|monthly|quarterly|yearly (opcional).
      limit: máximo de resultados, 1-50 (default 10).
    """
    body: dict = {"query_text": query_text, "limit": limit}
    if agent_identifier is not None:
        body["agent_identifier"] = agent_identifier
    if level is not None:
        body["level"] = level
    try:
        data = _api_call("POST", "/api/v1/clusters/search", json=body)
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def list_clusters(agent_identifier: str, level: Optional[str] = None,
                  status: str = "active", limit: int = 20) -> dict:
    """Listar clusters de memoria de un agente, filtrados por nivel y estado.

    Args:
      agent_identifier: nombre del agente (obligatorio).
      level: weekly|monthly|quarterly|yearly (opcional).
      status: candidate|active|rejected|superseded (default active).
      limit: máximo de resultados, 1-100 (default 20).
    """
    params: dict = {"agent_identifier": agent_identifier, "status": status,
                    "limit": limit}
    if level is not None:
        params["level"] = level
    try:
        data = _api_call("GET", "/api/v1/clusters", params=params)
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def read_cluster(cluster_id: str, include_members: bool = False,
                 include_sources: bool = False) -> dict:
    """Leer el detalle de un cluster, opcionalmente con miembros y fuentes.

    Args:
      cluster_id: UUID del cluster.
      include_members: si true, añade la lista de memorias miembro.
      include_sources: si true, añade fuentes y clusters padre (navegación telescópica).
    """
    if not _UUID_RE.match(cluster_id):
        return _err(RuntimeError("cluster_id must be a valid UUID"))
    cid = quote(str(cluster_id), safe="")
    try:
        data = _api_call("GET", f"/api/v1/clusters/{cid}")
        if include_members:
            data["members"] = _api_call("GET", f"/api/v1/clusters/{cid}/members")
        if include_sources:
            data["sources_detail"] = _api_call("GET", f"/api/v1/clusters/{cid}/sources")
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def get_briefing(agent_identifier: str) -> dict:
    """Briefing del agente: foresights, tensiones, clusters pendientes y resumen telescópico.

    Args:
      agent_identifier: nombre del agente (obligatorio).
    """
    try:
        data = _api_call("GET", "/api/v1/briefing",
                         params={"agent_identifier": agent_identifier})
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool(meta={"anthropic/maxResultSizeChars": 400000})
def get_telescopic_view(agent_identifier: str,
                        levels: str = "weekly,monthly,quarterly,yearly") -> dict:
    """Cargar la cadena de memoria fractal del agente para el protocolo de boot.

    Args:
      agent_identifier: nombre del agente (obligatorio).
      levels: niveles separados por comas (default "weekly,monthly,quarterly,yearly").
    """
    try:
        data = _api_call("GET", "/api/v1/clusters/telescopic",
                         params={"agent_identifier": agent_identifier, "levels": levels})
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


# anthropic/maxResultSizeChars: a full fractal (quarterly narratives 3-4K
# words + a month of weeklies + loose raw memories) exceeds Claude Code's
# default 25K-token MCP output cap. Annotating the size server-side lets the
# boot arrive in ONE call without any client-side env configuration.
@mcp.tool(meta={"anthropic/maxResultSizeChars": 400000})
def get_progressive_view(agent_identifier: str, max_recent_days: int = 14,
                         sections: str = "all") -> dict:
    """Vista telescópica progresiva — cada capa temporal comprime la anterior.

    Lo cerrado no se re-lee: yearly de años anteriores → quarterly no
    absorbidos por un yearly → monthly no absorbidos por un quarterly →
    weekly sueltos del mes abierto → días sueltos (memorias crudas no
    tejidas en ninguna semana). Cada cluster llega como unidad individual
    y el orden de lectura es exacto: yearly → quarterly → monthly →
    weekly → recent_days, de más antiguo a más reciente dentro de cada
    nivel. UNA llamada carga el boot completo.

    Args:
      agent_identifier: nombre del agente (obligatorio).
      max_recent_days: cap de días crudos al final, 1-31 (default 14).
      sections: "all" (default) o lista separada por comas de
        yearly,quarterly,monthly,weekly,recent_days para cargar la vista
        por capítulos si un cliente tiene límite de respuesta.
    """
    try:
        data = _api_call("GET", "/api/v1/clusters/telescopic/progressive",
                         params={"agent_identifier": agent_identifier,
                                 "max_recent_days": max_recent_days,
                                 "sections": sections})
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool(meta={"anthropic/maxResultSizeChars": 200000})
def fractal_search(agent_identifier: str, cluster_id: Optional[str] = None,
                   query_text: Optional[str] = None,
                   level: Optional[str] = None, limit: int = 20) -> dict:
    """Búsqueda fractal: navegar la memoria desde la máxima abstracción hacia abajo.

    Sin cluster_id entra por el nivel más alto que exista (yearly →
    quarterly → monthly → weekly). Con cluster_id devuelve sus hijos:
    clusters fuente si es un nivel alto, memorias concretas si es weekly.
    Cada hijo trae su id — para bajar otro nivel, repite la llamada con
    ese id como cluster_id. query_text opcional rankea los hijos por
    similitud semántica dentro del scope; sin query, orden cronológico.

    Args:
      agent_identifier: nombre del agente (obligatorio).
      cluster_id: UUID del cluster en el que hacer zoom (opcional).
      query_text: texto para rankear semánticamente, 3-2000 chars (opcional).
      level: nivel de entrada explícito weekly|monthly|quarterly|yearly (opcional).
      limit: máximo de hijos, 1-100 (default 20).
    """
    if cluster_id is not None and not _UUID_RE.match(cluster_id):
        return _err(RuntimeError("cluster_id must be a valid UUID"))
    body: dict = {"agent_identifier": agent_identifier, "limit": limit}
    if cluster_id is not None:
        body["cluster_id"] = cluster_id
    if query_text is not None:
        body["query_text"] = query_text
    if level is not None:
        body["level"] = level
    try:
        data = _api_call("POST", "/api/v1/clusters/zoom", json=body)
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


@mcp.tool()
def narrate_cluster(cluster_id: str, narrative: str) -> dict:
    """Escribir o actualizar la narrativa de un cluster (requiere ser dueño del agente).

    Args:
      cluster_id: UUID del cluster.
      narrative: texto de la narrativa (1-5000 chars).
    """
    if not _UUID_RE.match(cluster_id):
        return _err(RuntimeError("cluster_id must be a valid UUID"))
    cid = quote(str(cluster_id), safe="")
    try:
        data = _api_call("PUT", f"/api/v1/clusters/{cid}/narrate",
                         json={"narrative": narrative})
        return _ok(data)
    except RuntimeError as e:
        return _err(e)


# ---------------------------------------------------------------------------
# Entry point: stdio MCP server (Claude Code lo invoca como subprocess).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if MCP_TRANSPORT in ("sse", "streamable-http"):
        # Server por red — escucha en MCP_HOST:MCP_PORT.
        mcp.run(transport=MCP_TRANSPORT)
    else:
        # Default stdio — Claude Code lo invoca como subprocess.
        mcp.run()
