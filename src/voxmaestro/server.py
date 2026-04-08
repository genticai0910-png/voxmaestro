"""
VoxMaestro HTTP server.

Exposes VoxMaestro as a microservice on port 8850.
Intended for deployment on Mac Mini via LaunchAgent.

Endpoints:
  GET  /health          — liveness check
  GET  /diagram         — Mermaid state diagram
  POST /replay          — replay a Bland post-call transcript
  POST /score           — score a Bland payload (no Langfuse, fast)
  POST /live-turn       — Bland live-turn tool webhook (same as /voice/vox-turn)
  POST /analyze         — replay + iRELOP enrich in one shot
  GET  /analytics       — funnel report from in-memory call history
  POST /analytics/ingest — ingest a CallAnalysis into the in-memory analyzer

Env vars:
  VOX_API_KEY      — Bearer token (optional; if unset, auth is skipped)
  VOX_AGENT_YAML   — path to agent YAML (default: examples/real_estate_agent.yaml)
  VOX_PORT         — port (default: 8850)
  VOX_HOST         — host (default: 127.0.0.1)
  IRELOP_WEBHOOK_URL — n8n lead-enrich webhook
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, Security
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from .conductor import VoxMaestro
from .integrations.bland import BlandLiveTurnHandler, BlandTranscriptAdapter, qualification_score
from .integrations.irelop import VoxIRELOPEnricher
from .analytics import CallFunnelAnalyzer
from .diagram import generate_mermaid, generate_mermaid_html

logger = logging.getLogger(__name__)

# ── Global singletons ──────────────────────────────────────────────────────────

_conductor: Optional[VoxMaestro] = None
_adapter: Optional[BlandTranscriptAdapter] = None
_live_handler: Optional[BlandLiveTurnHandler] = None
_enricher = VoxIRELOPEnricher()
_analyzer = CallFunnelAnalyzer()
_http_client: Optional[httpx.AsyncClient] = None


def _find_yaml() -> Path:
    custom = os.environ.get("VOX_AGENT_YAML", "")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
    candidates = [
        Path.home() / "voxmaestro/examples/real_estate_agent.yaml",
        Path(__file__).parent.parent.parent / "examples/real_estate_agent.yaml",
        Path("examples/real_estate_agent.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not locate agent YAML. Set VOX_AGENT_YAML env var.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _conductor, _adapter, _live_handler, _http_client
    yaml_path = _find_yaml()
    logger.info("voxmaestro_server_start yaml=%s", yaml_path)
    _conductor = VoxMaestro.from_yaml(yaml_path)
    _http_client = httpx.AsyncClient(timeout=15.0)
    _conductor.set_http_client(_http_client)
    _adapter = BlandTranscriptAdapter(_conductor)
    _live_handler = BlandLiveTurnHandler(_conductor)
    yield
    if _http_client:
        await _http_client.aclose()
    if _conductor:
        await _conductor.close()
    logger.info("voxmaestro_server_stop")


if _HAS_FASTAPI:
    app = FastAPI(
        title="VoxMaestro",
        description="Voice agent conductor microservice",
        version="0.1.0",
        lifespan=lifespan,
    )

    _bearer = HTTPBearer(auto_error=False)

    def _check_auth(creds: Optional[HTTPAuthorizationCredentials] = Security(_bearer)):
        api_key = os.environ.get("VOX_API_KEY", "")
        if not api_key:
            return  # auth disabled
        if not creds or creds.credentials != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # ── Request models ─────────────────────────────────────────────────────────

    class ReplayRequest(BaseModel):
        call_id: Optional[str] = None
        transcript: Any = None          # list[dict] or str
        duration: Optional[float] = None
        metadata: dict = {}

    class LiveTurnRequest(BaseModel):
        call_id: Optional[str] = None
        transcript: Any = None
        variables: dict = {}

    class AnalyzeRequest(BaseModel):
        call_id: Optional[str] = None
        lead_id: Optional[str] = None
        transcript: Any = None
        duration: Optional[float] = None
        existing_lead_data: dict = {}

    # ── Endpoints ──────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "conductor": _conductor._agent.get("name") if _conductor else None,
            "states": len(_conductor._states) if _conductor else 0,
        }

    @app.get("/diagram", response_class=PlainTextResponse)
    async def diagram(_=Depends(_check_auth)):
        if not _conductor:
            raise HTTPException(503, "Conductor not initialized")
        return generate_mermaid(_conductor)

    @app.get("/diagram.html", response_class=HTMLResponse)
    async def diagram_html(_=Depends(_check_auth)):
        if not _conductor:
            raise HTTPException(503, "Conductor not initialized")
        return generate_mermaid_html(_conductor)

    @app.post("/replay", dependencies=[Depends(_check_auth)])
    async def replay(req: ReplayRequest):
        if not _adapter:
            raise HTTPException(503, "Adapter not initialized")
        payload = {
            "call_id": req.call_id or "unknown",
            "transcript": req.transcript or [],
            "duration": req.duration,
        }
        payload.update(req.metadata)
        t0 = time.time()
        analysis = await _adapter.replay(payload, call_id=req.call_id)
        score = qualification_score(analysis)
        latency = round(time.time() - t0, 3)
        _analyzer.ingest_one(analysis)
        return {
            "call_id": analysis.call_id,
            "final_state": analysis.final_state,
            "phase": analysis.phase,
            "score": score,
            "tier": "HOT" if score >= 80 else "WARM" if score >= 60 else "COOL" if score >= 40 else "PASS",
            "state_path": analysis.state_path,
            "intents": analysis.intents,
            "turns_processed": analysis.turns_processed,
            "qualification_reached": analysis.qualification_reached,
            "pricing_reached": analysis.pricing_reached,
            "offer_reached": analysis.offer_reached,
            "handoff_triggered": analysis.handoff_triggered,
            "latency_seconds": latency,
        }

    @app.post("/score", dependencies=[Depends(_check_auth)])
    async def score(req: ReplayRequest):
        """Fast score — same as replay but returns minimal response."""
        return await replay(req)

    @app.post("/live-turn")
    async def live_turn(request: Request):
        """Bland live-turn tool webhook — no auth (called by Bland directly)."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"response": ""})
        if not _live_handler:
            return JSONResponse({"response": ""})
        result = await _live_handler.handle(body)
        return JSONResponse(result)

    @app.post("/analyze", dependencies=[Depends(_check_auth)])
    async def analyze(req: AnalyzeRequest):
        """Replay + iRELOP enrich in one call."""
        if not _adapter:
            raise HTTPException(503, "Not initialized")
        payload = {
            "call_id": req.call_id or "unknown",
            "transcript": req.transcript or [],
            "duration": req.duration,
        }
        analysis = await _adapter.replay(payload, call_id=req.call_id)
        signals = _enricher.extract_signals(analysis)
        score = qualification_score(analysis)
        lead_data_patch = _enricher.to_lead_data_patch(signals, req.existing_lead_data or {})

        # Auto-fire iRELOP webhook if configured
        irelop_url = os.environ.get("IRELOP_WEBHOOK_URL", "")
        irelop_fired = False
        if irelop_url and req.lead_id:
            try:
                result = await _enricher.enrich_and_post(analysis, req.lead_id, irelop_url, req.existing_lead_data)
                irelop_fired = result.get("status") == "ok"
            except Exception as e:
                logger.warning("analyze_irelop_failed error=%s", e)

        _analyzer.ingest_one(analysis)

        return {
            "call_id": analysis.call_id,
            "score": score,
            "tier": signals.voice_tier,
            "final_state": analysis.final_state,
            "signals": {
                "motivation_bonus": signals.motivation_bonus,
                "timeline_days": signals.timeline_days,
                "distress_signals": signals.distress_signals,
                "voice_tier": signals.voice_tier,
                "voice_score": signals.voice_score,
            },
            "lead_data_patch": lead_data_patch,
            "irelop_fired": irelop_fired,
        }

    @app.get("/analytics", dependencies=[Depends(_check_auth)])
    async def analytics():
        import json
        return JSONResponse(content=json.loads(_analyzer.to_json()))
