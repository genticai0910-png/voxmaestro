"""
VoxMaestro — Voice Agent Conductor Engine
The open source orchestration layer for real-time voice AI agents.

Parses a VoxMaestro YAML config and provides:
  1. Conversation State Machine with guarded transitions
  2. Mid-Turn Tool Bridge with pre-LLM filler gating
  3. Human Handoff Protocol (3-phase)
  4. Intent → State routing via pluggable classifiers
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

logger = logging.getLogger("voxmaestro")


# ─── Data Models ────────────────────────────────────────────────────

class CallPhase(Enum):
    ACTIVE = auto()
    TOOL_PENDING = auto()
    FILLER_PLAYING = auto()
    HANDOFF_BRIDGE = auto()
    HANDOFF_TEARDOWN = auto()
    EXITED = auto()


@dataclass
class ConversationContext:
    """Mutable conversation state — lives in-process memory (hot path)."""

    call_id: str
    caller_phone: str = ""
    current_state: str = "initial"
    previous_state: str = "initial"
    phase: CallPhase = CallPhase.ACTIVE
    turn_count: int = 0
    state_turn_count: int = 0  # Turns within current state
    intent_history: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict] = field(default_factory=list)
    irelop_score: Optional[dict] = None
    handoff_reason: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return time.time() - self.start_time

    def add_turn(self, role: str, content: str, intent: Optional[str] = None):
        self.conversation_history.append({
            "role": role,
            "content": content,
            "intent": intent,
            "timestamp": time.time(),
        })
        self.turn_count += 1
        self.state_turn_count += 1
        if intent:
            self.intent_history.append(intent)


@dataclass
class TransitionResult:
    """Result of a state transition evaluation."""

    new_state: str
    tool_to_fire: Optional[str] = None
    filler: Optional[dict] = None
    trigger: Optional[str] = None  # "handoff", "graceful_exit", etc.
    blocked: bool = False
    block_reason: Optional[str] = None


@dataclass
class ToolCallResult:
    """Result from a mid-turn tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0


# ─── Schema Loader ──────────────────────────────────────────────────

class SchemaLoader:
    """Loads and validates a VoxMaestro YAML config."""

    @staticmethod
    def load(path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        SchemaLoader._validate(config)
        return config

    @staticmethod
    def _validate(config: dict):
        required = ["schema_version", "agent", "intent", "generation", "states"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

        # Validate state transitions reference valid states
        valid_states = set(config["states"].keys())
        for state_name, state_def in config["states"].items():
            transitions = state_def.get("transitions", {})
            for intent, target in transitions.items():
                if target not in valid_states and target not in ("previous",):
                    logger.warning(
                        f"State '{state_name}' transition '{intent}' → '{target}' "
                        f"references unknown state"
                    )

        # Validate intent → tool mappings
        tool_names = set(config.get("tools", {}).keys())
        for intent_def in config["intent"]["intents"]:
            tool = intent_def.get("tool")
            if tool and tool not in tool_names:
                logger.warning(
                    f"Intent '{intent_def['id']}' references unknown tool '{tool}'"
                )

        logger.info(
            f"Config validated: {len(valid_states)} states, "
            f"{len(config['intent']['intents'])} intents, "
            f"{len(tool_names)} tools"
        )


# ─── State Machine ──────────────────────────────────────────────────

class StateMachine:
    """
    Guarded state machine with intent-driven transitions.

    Static config defines WHAT transitions are legal.
    Intent classifier decides WHICH legal transition to take.
    """

    def __init__(self, config: dict):
        self.states = config["states"]
        self.tools = config.get("tools", {})
        self.intents = {i["id"]: i for i in config["intent"]["intents"]}

    def evaluate_transition(
        self, ctx: ConversationContext, intent: str
    ) -> TransitionResult:
        """
        Given current state + detected intent, determine the next state.
        Returns a TransitionResult with tool/filler/trigger info.
        """

        state_def = self.states.get(ctx.current_state)
        if not state_def:
            logger.error(f"Unknown state: {ctx.current_state}")
            return TransitionResult(new_state=ctx.current_state, blocked=True,
                                    block_reason=f"Unknown state: {ctx.current_state}")

        # Check max_turns guardrail
        max_turns = state_def.get("max_turns")
        if max_turns and ctx.state_turn_count >= max_turns:
            escalation = state_def.get("escalation", {})
            escalation_target = escalation.get("after_max_turns", "handoff")
            logger.info(
                f"Max turns ({max_turns}) reached in '{ctx.current_state}' → "
                f"escalating to '{escalation_target}'"
            )
            return TransitionResult(
                new_state=escalation_target,
                trigger="max_turns_escalation",
            )

        # Resolve transition target
        transitions = state_def.get("transitions", {})
        target = transitions.get(intent) or transitions.get("*")

        if not target:
            logger.debug(f"No transition for intent '{intent}' in state '{ctx.current_state}'")
            return TransitionResult(new_state=ctx.current_state)

        # Resolve "previous" for transient states (like tool_call)
        if target == "previous":
            target = ctx.previous_state

        # Check if intent has an associated tool
        intent_def = self.intents.get(intent, {})
        tool_name = intent_def.get("tool") if isinstance(intent_def, dict) else None
        trigger = intent_def.get("trigger") if isinstance(intent_def, dict) else None

        # If tool is needed, route through tool_call state
        filler = None
        if tool_name and tool_name in self.tools:
            tool_def = self.tools[tool_name]
            filler = tool_def.get("filler")
            target = "tool_call"

        # Override target if intent has a direct trigger
        if trigger:
            target = trigger if trigger in self.states else target

        return TransitionResult(
            new_state=target,
            tool_to_fire=tool_name,
            filler=filler,
            trigger=trigger,
        )

    def apply_transition(self, ctx: ConversationContext, result: TransitionResult):
        """Apply a transition result to the conversation context."""
        if result.blocked:
            return

        if result.new_state != ctx.current_state:
            ctx.previous_state = ctx.current_state
            ctx.current_state = result.new_state
            ctx.state_turn_count = 0
            logger.info(
                f"[{ctx.call_id}] State: {ctx.previous_state} → {ctx.current_state}"
            )


# ─── Tool Bridge ────────────────────────────────────────────────────

class ToolBridge:
    """
    Mid-turn async tool execution with pre-LLM filler gating.

    Flow:
      1. Intent triggers tool → filler plays IMMEDIATELY (pre-LLM gate)
      2. Tool call fires async
      3. Result injected into LLM context
      4. Generation resumes with tool result
    """

    def __init__(self, config: dict):
        self.tools = config.get("tools", {})
        self._http_client: Optional[Any] = None  # Injected at runtime

    async def execute(
        self,
        tool_name: str,
        ctx: ConversationContext,
        on_filler: Optional[Callable] = None,
    ) -> ToolCallResult:
        """
        Execute a tool call with filler gating.

        Args:
            tool_name: Which tool to fire
            ctx: Conversation context for param extraction
            on_filler: Callback to immediately play filler audio/text.
                       This fires BEFORE the tool call — pre-LLM gate.
        """

        tool_def = self.tools.get(tool_name)
        if not tool_def:
            return ToolCallResult(
                tool_name=tool_name, success=False,
                error=f"Unknown tool: {tool_name}"
            )

        # Phase 1: IMMEDIATELY fire filler (pre-LLM gate)
        ctx.phase = CallPhase.FILLER_PLAYING
        filler = tool_def.get("filler")
        if filler and on_filler:
            await on_filler(filler)

        # Phase 2: Fire tool call async
        ctx.phase = CallPhase.TOOL_PENDING
        start = time.monotonic()
        timeout_ms = tool_def.get("timeout_ms", 3000)

        try:
            result = await asyncio.wait_for(
                self._call_tool(tool_def, ctx),
                timeout=timeout_ms / 1000,
            )
            latency = (time.monotonic() - start) * 1000

            ctx.tool_results[tool_name] = result
            ctx.phase = CallPhase.ACTIVE

            logger.info(
                f"[{ctx.call_id}] Tool '{tool_name}' completed in {latency:.0f}ms"
            )

            return ToolCallResult(
                tool_name=tool_name, success=True,
                data=result, latency_ms=latency,
            )

        except asyncio.TimeoutError:
            latency = (time.monotonic() - start) * 1000
            ctx.phase = CallPhase.ACTIVE
            logger.warning(
                f"[{ctx.call_id}] Tool '{tool_name}' timed out after {latency:.0f}ms"
            )
            return ToolCallResult(
                tool_name=tool_name, success=False,
                error=f"Timeout after {timeout_ms}ms", latency_ms=latency,
            )

        except Exception as e:
            ctx.phase = CallPhase.ACTIVE
            logger.error(f"[{ctx.call_id}] Tool '{tool_name}' failed: {e}")
            return ToolCallResult(
                tool_name=tool_name, success=False, error=str(e)
            )

    async def _call_tool(self, tool_def: dict, ctx: ConversationContext) -> Any:
        """
        Make the actual HTTP call to the tool endpoint.
        Override this for custom transport (gRPC, IPC, etc.)
        """

        # Extract params from conversation context
        params = {}
        for param_key in tool_def.get("params_from_context", []):
            params[param_key] = ctx.metadata.get(param_key)

        # In production, this uses aiohttp/httpx
        # For now, return a structured placeholder
        endpoint = tool_def["endpoint"]
        method = tool_def["method"]

        if self._http_client:
            if method == "POST":
                resp = await self._http_client.post(endpoint, json=params)
            else:
                resp = await self._http_client.get(endpoint, params=params)
            return await resp.json()
        else:
            # Dry run mode — useful for testing state machine without live services
            logger.info(f"[DRY RUN] {method} {endpoint} params={params}")
            return {"dry_run": True, "tool": tool_def.get("endpoint")}


# ─── Handoff Protocol ──────────────────────────────────────────────

class HandoffProtocol:
    """
    Three-phase human handoff:
      Phase 1 (Decision)  — Log reason, capture state
      Phase 2 (Bridge)    — Play filler, fire context payload async
      Phase 3 (Teardown)  — Save transcript, flush to training data, close stream
    """

    def __init__(self, config: dict):
        self.handoff_config = config.get("handoff", {})
        self.states = config.get("states", {})

    async def execute(
        self,
        ctx: ConversationContext,
        on_filler: Optional[Callable] = None,
        on_transfer: Optional[Callable] = None,
    ) -> dict:
        """Execute the three-phase handoff protocol."""

        handoff_state = self.states.get("handoff", {})
        phases = handoff_state.get("phases", {})
        results = {"phases_completed": []}

        # ── Phase 1: Decision ──
        ctx.phase = CallPhase.HANDOFF_BRIDGE
        decision = phases.get("decision", {})
        if decision.get("capture_reason"):
            logger.info(
                f"[{ctx.call_id}] Handoff initiated. "
                f"Reason: {ctx.handoff_reason or 'caller_request'}"
            )
        results["phases_completed"].append("decision")
        results["reason"] = ctx.handoff_reason or "caller_request"

        # ── Phase 2: Bridge ──
        bridge = phases.get("bridge", {})
        filler_text = bridge.get("filler", "Let me connect you with someone.")
        if on_filler:
            await on_filler({"text": filler_text})

        # Fire context payload to all delivery channels
        payload = await self._build_payload(ctx)
        delivery_results = await self._deliver_payload(payload)
        results["phases_completed"].append("bridge")
        results["delivery"] = delivery_results

        # ── Phase 3: Teardown ──
        ctx.phase = CallPhase.HANDOFF_TEARDOWN
        teardown = phases.get("teardown", {})

        teardown_data = {
            "transcript": ctx.conversation_history if teardown.get("save_transcript") else None,
            "irelop_score": ctx.irelop_score if teardown.get("save_irelop_score") else None,
            "save_to": teardown.get("save_to"),
        }

        if on_transfer:
            await on_transfer(teardown_data)

        ctx.phase = CallPhase.EXITED
        results["phases_completed"].append("teardown")

        logger.info(
            f"[{ctx.call_id}] Handoff complete. Phases: {results['phases_completed']}"
        )
        return results

    async def _build_payload(self, ctx: ConversationContext) -> dict:
        """Build the context payload from config + conversation state."""
        payload_fields = self.handoff_config.get("payload", [])
        payload = {}

        field_map = {
            "caller_phone": ctx.caller_phone,
            "irelop_score": ctx.irelop_score.get("total") if ctx.irelop_score else None,
            "irelop_breakdown": ctx.irelop_score,
            "intent_history": ctx.intent_history,
            "qualification_progress": ctx.metadata.get("qualification_progress"),
            "conversation_summary": None,  # Generated by LLM at handoff time
            "handoff_reason": ctx.handoff_reason,
            "caller_sentiment": ctx.metadata.get("sentiment"),
            "call_duration_seconds": round(ctx.duration_seconds),
            "transcript_url": None,  # Set after transcript save
        }

        for f in payload_fields:
            payload[f] = field_map.get(f)

        return payload

    async def _deliver_payload(self, payload: dict) -> list[dict]:
        """Send payload to all configured delivery channels."""
        deliveries = self.handoff_config.get("delivery", [])
        results = []

        for delivery in deliveries:
            channel = delivery.get("channel", "unknown")
            try:
                # In production: aiohttp POST to webhook/Slack/etc.
                logger.info(f"Delivering handoff payload via {channel}")
                results.append({"channel": channel, "status": "sent"})
            except Exception as e:
                logger.error(f"Handoff delivery failed ({channel}): {e}")
                results.append({"channel": channel, "status": "failed", "error": str(e)})

        return results


# ─── Conductor (Main Orchestrator) ──────────────────────────────────

class VoxMaestro:
    """
    The conductor. Wires together state machine, tool bridge, and handoff
    into a single coherent voice agent orchestration layer.

    Usage:
        conductor = VoxMaestro.from_yaml("agent.yaml")
        ctx = conductor.new_call(call_id="abc-123", caller_phone="+15551234567")

        # On each caller utterance:
        result = await conductor.process_turn(ctx, caller_text="do you have Thursday at 3?")
        # result.response_text = "Let me check... Yes, we have 3pm available."
        # result.filler = {"text": "Let me check...", "audio": "fillers/checking.wav"}
    """

    def __init__(self, config: dict):
        self.config = config
        self.state_machine = StateMachine(config)
        self.tool_bridge = ToolBridge(config)
        self.handoff = HandoffProtocol(config)
        self.guardrails = config.get("guardrails", {})

        # Pluggable callbacks — set by framework integration (Pipecat, LiveKit, etc.)
        self.on_filler: Optional[Callable] = None
        self.on_transfer: Optional[Callable] = None
        self.on_metric: Optional[Callable] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VoxMaestro":
        config = SchemaLoader.load(path)
        return cls(config)

    def new_call(self, call_id: str, caller_phone: str = "", **kwargs) -> ConversationContext:
        """Create a new conversation context for an incoming call."""
        ctx = ConversationContext(
            call_id=call_id,
            caller_phone=caller_phone,
        )
        ctx.metadata.update(kwargs)
        logger.info(f"[{call_id}] New call started. Initial state: {ctx.current_state}")
        return ctx

    async def process_turn(
        self,
        ctx: ConversationContext,
        caller_text: str,
        intent: Optional[str] = None,
    ) -> dict:
        """
        Process a single conversation turn.

        Args:
            ctx: The mutable conversation context
            caller_text: What the caller said (transcribed)
            intent: Pre-classified intent. If None, conductor will classify.

        Returns:
            dict with keys: response_text, filler, tool_result, state, action
        """

        result = {
            "response_text": None,
            "filler": None,
            "tool_result": None,
            "state": ctx.current_state,
            "action": None,
        }

        # ── Guardrail: max call duration ──
        max_duration = self.guardrails.get("max_call_duration_seconds", 600)
        if ctx.duration_seconds > max_duration:
            ctx.handoff_reason = "max_duration_exceeded"
            result["action"] = "handoff"
            result["response_text"] = "I appreciate your time. Let me connect you with someone who can continue helping you."
            await self.handoff.execute(ctx, on_filler=self.on_filler, on_transfer=self.on_transfer)
            return result

        # ── Record caller turn ──
        ctx.add_turn("caller", caller_text, intent=intent)

        # ── Classify intent (if not pre-classified) ──
        if not intent:
            # In production, this calls VSAI via HTTP
            # Conductor is classifier-agnostic — just needs an intent string back
            intent = await self._classify_intent(caller_text, ctx)

        # ── Evaluate state transition ──
        transition = self.state_machine.evaluate_transition(ctx, intent)

        # ── Handle tool calls (mid-turn bridge) ──
        if transition.tool_to_fire:
            result["filler"] = transition.filler

            tool_result = await self.tool_bridge.execute(
                transition.tool_to_fire, ctx, on_filler=self.on_filler
            )
            result["tool_result"] = tool_result

            if not tool_result.success:
                # Tool failed — use failure config
                tool_def = self.tool_bridge.tools.get(transition.tool_to_fire, {})
                failure = tool_def.get("on_failure", {})
                result["response_text"] = failure.get("message", "I'm sorry, I'm having trouble with that.")
                if failure.get("trigger") == "handoff":
                    result["action"] = "handoff"

            # Emit metric
            if self.on_metric:
                await self.on_metric("tool_call_latency_ms", tool_result.latency_ms,
                                     {"tool": transition.tool_to_fire, "success": tool_result.success})

        # ── Handle handoff trigger ──
        if transition.trigger == "handoff" or result.get("action") == "handoff":
            ctx.handoff_reason = ctx.handoff_reason or f"intent:{intent}"
            handoff_result = await self.handoff.execute(
                ctx, on_filler=self.on_filler, on_transfer=self.on_transfer
            )
            result["action"] = "handoff"
            result["handoff"] = handoff_result

        # ── Handle graceful exit ──
        elif transition.trigger == "graceful_exit":
            exit_state = self.config["states"].get("exit", {})
            result["response_text"] = exit_state.get("farewell_message", "Goodbye!")
            result["action"] = "exit"
            ctx.phase = CallPhase.EXITED

        # ── Apply state transition ──
        self.state_machine.apply_transition(ctx, transition)
        result["state"] = ctx.current_state

        return result

    async def _classify_intent(self, text: str, ctx: ConversationContext) -> str:
        """
        Classify caller intent. Override this with your VSAI model integration.
        Default implementation returns 'unknown'.
        """
        # In production: HTTP call to VSAI endpoint
        # intent_config = self.config["intent"]
        # resp = await http_client.post(intent_config["endpoint"], json={...})
        # return resp["intent"]
        return "unknown"

    async def handle_barge_in(self, ctx: ConversationContext) -> dict:
        """
        Handle caller interruption. Called by the audio pipeline
        when barge-in is detected.

        This is the HOT PATH — must complete in <100ms.
        No serialization, no persistence, no network calls.
        """
        barge_config = self.guardrails.get("barge_in", {})
        if not barge_config.get("enabled", True):
            return {"action": "ignore"}

        action = barge_config.get("action", "cancel_tts_resume_stt")

        if self.on_metric:
            await self.on_metric("barge_in_count", 1, {"state": ctx.current_state})

        logger.debug(f"[{ctx.call_id}] Barge-in detected. Action: {action}")

        return {
            "action": action,
            "cancel_tts": True,
            "resume_stt": True,
            "flush_audio_buffer": True,
        }

    async def handle_silence(self, ctx: ConversationContext) -> dict:
        """Handle prolonged silence from caller."""
        silence_prompt = self.guardrails.get("silence_prompt", "Are you still there?")
        return {
            "action": "prompt",
            "response_text": silence_prompt,
        }
