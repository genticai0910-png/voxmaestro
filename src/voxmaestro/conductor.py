"""
VoxMaestro conductor — YAML-defined voice agent state machine.

Fixes applied (see CHANGELOG):
  F1  Auto-return from tool_call state via return_to field
  F2  Per-call callbacks on ConversationContext (not VoxMaestro instance)
  F4  Turn recorded before max-duration check
  F5  apply_transition skipped after handoff/exit
  F6  HTTP client injection + lifecycle
  F10 JSON Schema validation on load
  F13 Tool call retry with exponential backoff
"""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import httpx
import yaml

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

try:
    from langfuse import Langfuse
    _HAS_LANGFUSE = True
except ImportError:
    _HAS_LANGFUSE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConversationPhase(str, Enum):
    ACTIVE = "active"
    EXITED = "exited"
    TRANSFERRED = "transferred"
    TIMED_OUT = "timed_out"
    ERROR = "error"


class IntentResult(str, Enum):
    CLASSIFIED = "classified"
    FALLBACK = "fallback"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str  # "user" | "agent"
    text: str
    intent: Optional[str] = None
    state: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class HandoffPayload:
    call_id: str
    reason: str
    transcript: list[Turn]
    final_state: str
    phase: ConversationPhase
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Per-call state container. All callbacks live here (Fix F2)."""

    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    current_state: str = "greeting"
    previous_state: Optional[str] = None
    phase: ConversationPhase = ConversationPhase.ACTIVE
    turn_count: int = 0
    state_turn_count: int = 0
    start_time: float = field(default_factory=time.time)
    transcript: list[Turn] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Per-call callbacks (F2) — set by VoxMaestro from defaults, can be overridden
    on_filler: Optional[Callable[[str], Coroutine]] = None
    on_transfer: Optional[Callable[[HandoffPayload], Coroutine]] = None
    on_metric: Optional[Callable[[str, Any], Coroutine]] = None

    def add_turn(self, role: str, text: str, intent: Optional[str] = None, **meta) -> Turn:
        t = Turn(role=role, text=text, intent=intent, state=self.current_state, metadata=meta)
        self.transcript.append(t)
        if role == "user":
            self.turn_count += 1
            self.state_turn_count += 1
        return t

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time


# ---------------------------------------------------------------------------
# Schema loader
# ---------------------------------------------------------------------------

class SchemaLoader:
    """Loads and validates agent YAML config against bundled JSON Schema (F10)."""

    _schema: Optional[dict] = None

    @classmethod
    def _get_schema(cls) -> Optional[dict]:
        if cls._schema is not None:
            return cls._schema
        try:
            schema_path = Path(__file__).parent / "schema.json"
            if schema_path.exists():
                cls._schema = json.loads(schema_path.read_text())
                return cls._schema
        except Exception as e:
            logger.warning("Could not load schema.json: %s", e)
        return None

    @classmethod
    def load(cls, path: str | Path) -> dict:
        raw = Path(path).read_text()
        config = yaml.safe_load(raw)

        if _HAS_JSONSCHEMA:
            schema = cls._get_schema()
            if schema:
                try:
                    jsonschema.validate(instance=config, schema=schema)
                    logger.debug("YAML config validated against schema")
                except jsonschema.ValidationError as e:
                    raise ValueError(f"Agent config validation failed: {e.message}") from e
        else:
            logger.debug("jsonschema not installed — skipping config validation")

        return config


# ---------------------------------------------------------------------------
# Tool Bridge
# ---------------------------------------------------------------------------

class ToolBridge:
    """Executes external tool calls over HTTP with retry + backoff (F6, F13)."""

    def __init__(self, tools_config: dict[str, dict]):
        self._tools = tools_config or {}
        self._client: Optional[httpx.AsyncClient] = None
        self._owns_client = False

    def set_http_client(self, client: httpx.AsyncClient) -> None:
        """Inject an external httpx client (F6). Bridge will NOT close it."""
        self._client = client
        self._owns_client = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        """Close the HTTP client if we own it (F6)."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def call(
        self,
        tool_name: str,
        ctx: ConversationContext,
        extra_payload: Optional[dict] = None,
    ) -> dict:
        """Execute a tool call with retry/backoff (F13)."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_cfg = self._tools[tool_name]
        endpoint = tool_cfg.get("endpoint", "")
        method = tool_cfg.get("method", "POST").upper()
        timeout = tool_cfg.get("timeout", 10.0)
        max_retries = tool_cfg.get("retry", 2)

        payload = dict(tool_cfg.get("payload_template", {}))
        payload["call_id"] = ctx.call_id
        payload["state"] = ctx.current_state
        if extra_payload:
            payload.update(extra_payload)

        client = await self._get_client()
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                backoff = 0.5 * (2 ** (attempt - 1))
                logger.debug("Tool %s retry %d after %.1fs", tool_name, attempt, backoff)
                await asyncio.sleep(backoff)

            try:
                t0 = time.time()
                if method == "POST":
                    resp = await client.post(endpoint, json=payload, timeout=timeout)
                else:
                    resp = await client.get(endpoint, params=payload, timeout=timeout)
                resp.raise_for_status()
                latency = time.time() - t0
                logger.debug("Tool %s completed in %.3fs", tool_name, latency)
                return resp.json()

            except httpx.TimeoutException as e:
                last_exc = e
                logger.warning("Tool %s timeout on attempt %d", tool_name, attempt + 1)
            except httpx.HTTPStatusError as e:
                last_exc = e
                logger.warning("Tool %s HTTP %d on attempt %d", tool_name, e.response.status_code, attempt + 1)
                # Don't retry client errors (4xx)
                if e.response.status_code < 500:
                    break
            except Exception as e:
                last_exc = e
                logger.warning("Tool %s error on attempt %d: %s", tool_name, attempt + 1, e)

        raise RuntimeError(f"Tool {tool_name} failed after {max_retries + 1} attempts: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# Intent Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Classifies user utterance intent via Ollama/OpenAI-compatible endpoint."""

    def __init__(self, intent_config: dict):
        self._cfg = intent_config or {}
        self._client: Optional[httpx.AsyncClient] = None
        self._owns_client = False

    def set_http_client(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self._owns_client = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._cfg.get("timeout", 5.0))
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def classify(self, text: str, valid_intents: list[str], ctx: ConversationContext) -> str:
        """Return an intent string. Falls back to config fallback_intent on error."""
        fallback = self._cfg.get("fallback_intent", "general_response")
        endpoint = self._cfg.get("endpoint", "")
        model = self._cfg.get("model", "gemma4:e4b")
        provider = self._cfg.get("provider", "ollama")

        if not endpoint or not text.strip():
            return fallback

        intents_list = ", ".join(valid_intents)
        prompt = (
            f"Classify the following user utterance into exactly one of these intents: {intents_list}\n\n"
            f"Utterance: {text}\n\n"
            f"Respond with ONLY the intent name, nothing else."
        )

        try:
            client = await self._get_client()
            t0 = time.time()

            if provider == "ollama":
                resp = await client.post(
                    f"{endpoint}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                    timeout=self._cfg.get("timeout", 5.0),
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("message", {}).get("content", "").strip().lower()

                latency = time.time() - t0
                logger.debug("Intent classified as '%s' in %.3fs", raw, latency)

                # Match against valid intents (fuzzy: case-insensitive substring)
                for intent in valid_intents:
                    if intent.lower() in raw:
                        return intent

                logger.debug("Intent '%s' not in valid list, using fallback", raw)
                return fallback

            elif provider == "dealiq-ce":
                # CE model: dealiq-ce-v4b-mlx — Conversational Extraction
                # Prompts for structured intent extraction with RE-specific slots
                ce_prompt = (
                    f"You are analyzing a real estate seller conversation.\n"
                    f"Classify the seller's intent from: {intents_list}\n\n"
                    f"Utterance: {text}\n\n"
                    f"Respond with JSON only: {{\"intent\": \"...\", \"confidence\": 0.0-1.0}}"
                )
                resp = await client.post(
                    f"{endpoint}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": ce_prompt}],
                        "stream": False,
                    },
                    timeout=self._cfg.get("timeout", 5.0),
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("message", {}).get("content", "").strip()
                # Parse JSON from response
                import json as _json
                try:
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    if start >= 0 and end > start:
                        parsed = _json.loads(raw[start:end])
                        classified = parsed.get("intent", "").lower()
                        # Match to valid intents
                        for intent in valid_intents:
                            if intent.lower() == classified or intent.lower() in classified:
                                return intent
                except Exception:
                    pass
                return fallback

            else:
                # OpenAI-compatible
                resp = await client.post(
                    f"{endpoint}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 20,
                    },
                    timeout=self._cfg.get("timeout", 5.0),
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip().lower()

                latency = time.time() - t0
                logger.debug("Intent classified as '%s' in %.3fs", raw, latency)

                # Match against valid intents (fuzzy: case-insensitive substring)
                for intent in valid_intents:
                    if intent.lower() in raw:
                        return intent

                logger.debug("Intent '%s' not in valid list, using fallback", raw)
                return fallback

        except Exception as e:
            logger.warning("Intent classification failed: %s — using fallback", e)
            return fallback


# ---------------------------------------------------------------------------
# Core conductor
# ---------------------------------------------------------------------------

class VoxMaestro:
    """
    YAML-defined voice agent state machine conductor.

    Usage::

        conductor = VoxMaestro.from_yaml("examples/real_estate_agent.yaml")
        ctx = conductor.create_context()
        result = await conductor.process_turn(ctx, "Hey, I'm calling about my house")
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._agent = config.get("agent", {})
        self._states: dict[str, dict] = config.get("states", {})
        self._tools_cfg: dict[str, dict] = config.get("tools", {})
        self._handoff_cfg: dict = config.get("handoff", {})
        self._intent_cfg: dict = config.get("intent", {})
        self._obs_cfg: dict = config.get("observability", {})

        self._tool_bridge = ToolBridge(self._tools_cfg)
        self._intent_classifier = IntentClassifier(self._intent_cfg)

        # Default callbacks (F2) — stored here, copied to each new context
        self._default_on_filler: Optional[Callable] = None
        self._default_on_transfer: Optional[Callable] = None
        self._default_on_metric: Optional[Callable] = None

        # Langfuse (optional)
        self._langfuse: Optional[Any] = None
        if _HAS_LANGFUSE and self._obs_cfg.get("langfuse", {}).get("enabled"):
            try:
                lf_cfg = self._obs_cfg["langfuse"]
                self._langfuse = Langfuse(
                    public_key=lf_cfg.get("public_key", ""),
                    secret_key=lf_cfg.get("secret_key", ""),
                    host=lf_cfg.get("host", "https://cloud.langfuse.com"),
                )
            except Exception as e:
                logger.warning("Langfuse init failed: %s", e)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VoxMaestro":
        config = SchemaLoader.load(path)
        return cls(config)

    # ------------------------------------------------------------------
    # HTTP client injection (F6)
    # ------------------------------------------------------------------

    def set_http_client(self, client: httpx.AsyncClient) -> None:
        """Inject shared httpx client into tool bridge and intent classifier."""
        self._tool_bridge.set_http_client(client)
        self._intent_classifier.set_http_client(client)

    async def close(self) -> None:
        """Release resources."""
        await self._tool_bridge.close()
        await self._intent_classifier.close()

    # ------------------------------------------------------------------
    # Callbacks (F2)
    # ------------------------------------------------------------------

    def on_filler(self, fn: Callable) -> None:
        self._default_on_filler = fn

    def on_transfer(self, fn: Callable) -> None:
        self._default_on_transfer = fn

    def on_metric(self, fn: Callable) -> None:
        self._default_on_metric = fn

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def create_context(
        self,
        call_id: Optional[str] = None,
        initial_state: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ConversationContext:
        """Create a new per-call context with default callbacks copied in (F2)."""
        ctx = ConversationContext(
            call_id=call_id or str(uuid.uuid4()),
            current_state=initial_state or list(self._states.keys())[0],
            metadata=metadata or {},
        )
        # Copy default callbacks into context (F2 — per-call, not shared)
        ctx.on_filler = self._default_on_filler
        ctx.on_transfer = self._default_on_transfer
        ctx.on_metric = self._default_on_metric
        return ctx

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------

    def _get_state(self, name: str) -> Optional[dict]:
        return self._states.get(name)

    def _get_valid_intents(self, state_name: str) -> list[str]:
        state = self._get_state(state_name)
        if not state:
            return []
        return list(state.get("transitions", {}).keys())

    def _apply_transition(self, ctx: ConversationContext, intent: str) -> bool:
        """Transition to next state based on intent. Returns True if transitioned."""
        state = self._get_state(ctx.current_state)
        if not state:
            logger.warning("Unknown state '%s' — blocked", ctx.current_state)
            return False

        transitions = state.get("transitions", {})
        next_state = transitions.get(intent) or transitions.get("*")

        if not next_state:
            logger.debug("No transition for intent '%s' in state '%s'", intent, ctx.current_state)
            return False

        if next_state not in self._states:
            logger.warning("Transition target '%s' does not exist in states", next_state)
            return False

        is_self_loop = next_state == ctx.current_state
        ctx.previous_state = ctx.current_state
        ctx.current_state = next_state
        # Don't reset state_turn_count on self-loops — max_turns needs to accumulate
        if not is_self_loop:
            ctx.state_turn_count = 0
        logger.debug("Transition: %s -[%s]-> %s", ctx.previous_state, intent, ctx.current_state)
        return True

    def _check_max_turns(self, ctx: ConversationContext) -> bool:
        """Returns True if state has exceeded max_turns."""
        state = self._get_state(ctx.current_state)
        if not state:
            return False
        max_turns = state.get("max_turns")
        if max_turns is None:
            return False
        return ctx.state_turn_count >= max_turns

    def _check_max_duration(self, ctx: ConversationContext) -> bool:
        """Returns True if call has exceeded max_duration_seconds."""
        max_dur = self._agent.get("max_duration_seconds")
        if not max_dur:
            return False
        return ctx.elapsed_seconds() >= max_dur

    # ------------------------------------------------------------------
    # Handoff / payload delivery
    # ------------------------------------------------------------------

    async def _deliver_payload(self, payload: HandoffPayload, ctx: ConversationContext) -> None:
        """POST handoff payload to configured webhook channels."""
        channels = self._handoff_cfg.get("channels", [])
        if not channels:
            logger.debug("No handoff channels configured")
            return

        payload_data = {
            "call_id": payload.call_id,
            "reason": payload.reason,
            "final_state": payload.final_state,
            "phase": payload.phase.value,
            "transcript": [
                {
                    "role": t.role,
                    "text": t.text,
                    "intent": t.intent,
                    "state": t.state,
                    "timestamp": t.timestamp,
                }
                for t in payload.transcript
            ],
            "metadata": payload.metadata,
        }

        client = await self._tool_bridge._get_client()
        for ch in channels:
            url = ch.get("webhook_url", "")
            if not url:
                continue
            # Env var substitution
            import os
            for k, v in os.environ.items():
                url = url.replace(f"${{{k}}}", v)

            try:
                resp = await client.post(url, json=payload_data, timeout=10.0)
                resp.raise_for_status()
                logger.info("Handoff delivered to %s", url)
            except Exception as e:
                logger.error("Handoff delivery failed for %s: %s", url, e)

    async def _trigger_handoff(self, ctx: ConversationContext, reason: str) -> HandoffPayload:
        """Finalize context and fire handoff callbacks."""
        ctx.phase = ConversationPhase.TRANSFERRED

        payload = HandoffPayload(
            call_id=ctx.call_id,
            reason=reason,
            transcript=list(ctx.transcript),
            final_state=ctx.current_state,
            phase=ctx.phase,
            metadata=ctx.metadata,
        )

        await self._deliver_payload(payload, ctx)

        if ctx.on_transfer:
            try:
                await ctx.on_transfer(payload)
            except Exception as e:
                logger.error("on_transfer callback error: %s", e)

        return payload

    async def _graceful_exit(self, ctx: ConversationContext, reason: str) -> None:
        """Exit the conversation gracefully (max turns / max duration)."""
        ctx.phase = ConversationPhase.TIMED_OUT if "duration" in reason else ConversationPhase.EXITED
        payload = HandoffPayload(
            call_id=ctx.call_id,
            reason=reason,
            transcript=list(ctx.transcript),
            final_state=ctx.current_state,
            phase=ctx.phase,
            metadata=ctx.metadata,
        )
        await self._deliver_payload(payload, ctx)

        if ctx.on_transfer:
            try:
                await ctx.on_transfer(payload)
            except Exception as e:
                logger.error("on_transfer callback error on graceful exit: %s", e)

    # ------------------------------------------------------------------
    # Main process_turn
    # ------------------------------------------------------------------

    async def process_turn(
        self,
        ctx: ConversationContext,
        user_text: str,
        pre_classified_intent: Optional[str] = None,
    ) -> dict:
        """
        Process a single user turn.

        Returns a dict with keys:
          - response_text: str
          - intent: str
          - state: str (current state after processing)
          - phase: str
          - handoff: Optional[dict]
          - filler: Optional[str]
          - tool_result: Optional[dict]
        """
        if ctx.phase != ConversationPhase.ACTIVE:
            return {
                "response_text": "",
                "intent": None,
                "state": ctx.current_state,
                "phase": ctx.phase.value,
                "handoff": None,
                "filler": None,
                "tool_result": None,
                "blocked": True,
                "blocked_reason": f"Context is {ctx.phase.value}",
            }

        result: dict = {
            "response_text": "",
            "intent": None,
            "state": ctx.current_state,
            "phase": ctx.phase.value,
            "handoff": None,
            "filler": None,
            "tool_result": None,
            "blocked": False,
        }

        # Langfuse trace
        trace = None
        if self._langfuse:
            try:
                trace = self._langfuse.trace(
                    name=f"vox-{self._agent.get('name', 'agent')}-{ctx.call_id}",
                    metadata={"state": ctx.current_state, "turn": ctx.turn_count},
                )
            except Exception:
                pass

        # F4: Record turn BEFORE max-duration check
        turn = ctx.add_turn("user", user_text)

        # Max duration check (F4: after turn recorded)
        if self._check_max_duration(ctx):
            await self._graceful_exit(ctx, "max_duration_exceeded")
            result.update({
                "phase": ctx.phase.value,
                "blocked": True,
                "blocked_reason": "max_duration_exceeded",
                "handoff": {"reason": "max_duration_exceeded"},
            })
            return result

        # --- Intent classification ---
        state_cfg = self._get_state(ctx.current_state)
        if not state_cfg:
            result.update({
                "blocked": True,
                "blocked_reason": f"unknown_state:{ctx.current_state}",
            })
            return result

        valid_intents = self._get_valid_intents(ctx.current_state)

        if pre_classified_intent:
            intent = pre_classified_intent
        elif valid_intents:
            span = None
            if trace:
                try:
                    span = trace.span(name="intent-classify")
                except Exception:
                    pass
            intent = await self._intent_classifier.classify(user_text, valid_intents, ctx)
            if span:
                try:
                    span.end(output={"intent": intent})
                except Exception:
                    pass
        else:
            intent = self._intent_cfg.get("fallback_intent", "general_response")

        turn.intent = intent
        result["intent"] = intent

        # Track metrics
        if ctx.on_metric:
            try:
                await ctx.on_metric("intent_classified", {"intent": intent, "state": ctx.current_state})
            except Exception:
                pass

        # --- Check max turns (before transition) ---
        if self._check_max_turns(ctx):
            # Look for escalation intent
            transitions = state_cfg.get("transitions", {})
            escalation_target = transitions.get("escalate") or transitions.get("max_turns_exceeded")
            if escalation_target and escalation_target in self._states:
                ctx.previous_state = ctx.current_state
                ctx.current_state = escalation_target
                ctx.state_turn_count = 0
                logger.info("Max turns in '%s' — escalating to '%s'", ctx.previous_state, ctx.current_state)
            else:
                await self._graceful_exit(ctx, f"max_turns_in_{ctx.current_state}")
                result.update({
                    "phase": ctx.phase.value,
                    "blocked": True,
                    "blocked_reason": "max_turns_exceeded",
                    "handoff": {"reason": f"max_turns_in_{ctx.current_state}"},
                })
                return result

        # --- Handoff intent check ---
        handoff_intents = {"transfer", "handoff", "escalate_live", "speak_to_human"}
        if intent in handoff_intents:
            handoff_payload = await self._trigger_handoff(ctx, f"user_requested_{intent}")
            result.update({
                "phase": ctx.phase.value,
                "handoff": {
                    "reason": handoff_payload.reason,
                    "call_id": handoff_payload.call_id,
                },
            })
            return result  # F5: early return after handoff, skip apply_transition

        # --- Filler audio ---
        filler = state_cfg.get("filler")
        if filler and ctx.on_filler:
            result["filler"] = filler
            try:
                await ctx.on_filler(filler)
            except Exception as e:
                logger.warning("on_filler callback error: %s", e)

        # --- Tool call ---
        tool_names = state_cfg.get("tools", [])
        tool_result = None
        if tool_names:
            tool_name = tool_names[0] if isinstance(tool_names[0], str) else tool_names[0].get("name")
            span = None
            if trace:
                try:
                    span = trace.span(name=f"tool-{tool_name}")
                except Exception:
                    pass
            try:
                tool_result = await self._tool_bridge.call(tool_name, ctx)
                result["tool_result"] = tool_result
            except Exception as e:
                logger.error("Tool call failed: %s", e)
                result["tool_result"] = {"error": str(e)}
                tool_result = {"error": str(e)}
            finally:
                if span:
                    try:
                        span.end(output={"result": tool_result})
                    except Exception:
                        pass

            # F1: Auto-return from tool_call state
            return_to = state_cfg.get("return_to")
            if return_to == "previous" and ctx.previous_state:
                logger.debug("Auto-returning from tool_call to '%s'", ctx.previous_state)
                new_current = ctx.previous_state
                ctx.previous_state = ctx.current_state
                ctx.current_state = new_current
                ctx.state_turn_count = 0
                result["state"] = ctx.current_state
                result["phase"] = ctx.phase.value
                return result
            elif return_to and return_to != "previous" and return_to in self._states:
                logger.debug("Auto-returning from tool_call to explicit state '%s'", return_to)
                ctx.previous_state = ctx.current_state
                ctx.current_state = return_to
                ctx.state_turn_count = 0
                result["state"] = ctx.current_state
                result["phase"] = ctx.phase.value
                return result

        # --- Apply state transition ---
        self._apply_transition(ctx, intent)  # F5: only reached if not exited/transferred

        result["state"] = ctx.current_state
        result["phase"] = ctx.phase.value

        # Build response text from new state's prompt
        new_state_cfg = self._get_state(ctx.current_state)
        if new_state_cfg:
            result["response_text"] = new_state_cfg.get("prompt", "")

        if trace:
            try:
                trace.update(output={"final_state": ctx.current_state, "intent": intent})
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # Transcript replay
    # ------------------------------------------------------------------

    async def transcript_replay(
        self,
        turns: list[str],
        call_id: Optional[str] = None,
        pre_classified_intents: Optional[list[str]] = None,
    ) -> dict:
        """
        Replay a list of user utterance strings through the state machine.
        Returns dict with state_path, intents, final_state, phase, turn_results.
        """
        ctx = self.create_context(call_id=call_id)
        state_path = [ctx.current_state]
        intent_path: list[Optional[str]] = []
        turn_results = []

        for i, text in enumerate(turns):
            if ctx.phase.value != "active":
                break
            pre = (
                pre_classified_intents[i]
                if pre_classified_intents and i < len(pre_classified_intents)
                else None
            )
            result = await self.process_turn(ctx, text, pre_classified_intent=pre)
            intent_path.append(result.get("intent"))
            state_path.append(ctx.current_state)
            turn_results.append(result)

        return {
            "call_id": ctx.call_id,
            "state_path": state_path,
            "intents": intent_path,
            "final_state": ctx.current_state,
            "phase": ctx.phase.value,
            "turn_results": turn_results,
            "transcript": [
                {"role": t.role, "text": t.text, "intent": t.intent}
                for t in ctx.transcript
            ],
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        return {
            "agent": self._agent,
            "states": list(self._states.keys()),
            "tools": list(self._tools_cfg.keys()),
        }
