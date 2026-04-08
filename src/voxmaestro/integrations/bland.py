"""
VoxMaestro Bland AI integration.

Provides post-call transcript replay: takes Bland's post-call webhook payload,
parses the transcript into turns, replays them through a VoxMaestro conductor,
and returns a structured call analysis result.

Also provides BlandLiveTurnHandler for real-time mid-call context injection.

Architecture: Bland → post-call webhook → BlandTranscriptAdapter.replay() → CallAnalysis
             Bland → live-turn tool webhook → BlandLiveTurnHandler.handle() → {"response": "..."}
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..conductor import VoxMaestro, ConversationContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CallAnalysis:
    call_id: str
    bland_call_id: str
    final_state: str
    phase: str
    state_path: list[str]           # sequence of states visited
    intents: list[str]              # sequence of intents classified
    turns_processed: int
    qualification_reached: bool     # whether "qualification" state was reached
    pricing_reached: bool           # whether "pricing_discussion" was reached
    offer_reached: bool             # whether "offer_preparation" or "offer_presentation" reached
    handoff_triggered: bool
    handoff_reason: Optional[str]
    transcript_turns: int           # total turns in original transcript
    duration_seconds: Optional[float]
    metadata: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_OFFER_STATES = {"offer_preparation", "offer_presentation", "negotiation"}
_CLOSING_STATES = {"closing", "appointment_setting", "wrap_up"}


def qualification_score(analysis: CallAnalysis) -> int:
    """
    Map call depth to 0-100 score based on states reached.
    """
    if analysis.handoff_triggered:
        return 100
    states = set(analysis.state_path)
    if states & _CLOSING_STATES:
        return 90
    if analysis.offer_reached:
        return 75
    if analysis.pricing_reached:
        return 50
    if analysis.qualification_reached:
        return 25
    return 0


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BlandTranscriptAdapter:
    """
    Replays a Bland post-call transcript through a VoxMaestro conductor.

    Post-call only — not real-time. Scores/tracks state machine progression
    from the completed call transcript.
    """

    def __init__(self, conductor: "VoxMaestro"):
        self._conductor = conductor

    def parse_transcript(self, bland_payload: dict) -> list[str]:
        """
        Extract user utterances from a Bland post-call payload.

        Bland sends one of:
          a) `transcript`: list of {"role": "user"|"assistant", "text": "...", ...}
          b) `concatenated_transcript`: flat string — cannot be reliably split, returns []

        Returns a list of user utterance strings (assistant turns are filtered out).
        """
        transcript = bland_payload.get("transcript")

        # Structured transcript (preferred)
        if isinstance(transcript, list):
            turns = []
            for item in transcript:
                if not isinstance(item, dict):
                    continue
                role = item.get("role", "")
                text = item.get("text", "").strip()
                if role == "user" and text:
                    turns.append(text)
            return turns

        # Flat string — can't reliably parse turn-by-turn
        if isinstance(transcript, str) and transcript.strip():
            logger.debug("bland_flat_transcript: skipping replay, returning empty")
            return []

        # Check concatenated_transcript fallback
        concat = bland_payload.get("concatenated_transcript", "")
        if concat and isinstance(concat, str):
            logger.debug("bland_concatenated_transcript: skipping replay, returning empty")
            return []

        return []

    async def replay(
        self,
        bland_payload: dict,
        call_id: Optional[str] = None,
        pre_classified_intents: Optional[list[str]] = None,
    ) -> CallAnalysis:
        """
        Replay a Bland post-call payload through the VoxMaestro state machine.

        Args:
            bland_payload: Raw Bland webhook payload dict.
            call_id: Optional override for internal call ID (defaults to UUID4).
            pre_classified_intents: Optional list of pre-classified intents,
                one per user turn. Skips live classification when provided.

        Returns:
            CallAnalysis with full state machine trace.
        """
        bland_call_id = bland_payload.get("call_id", bland_payload.get("id", "unknown"))
        duration = bland_payload.get("duration", bland_payload.get("call_duration"))
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = None

        user_turns = self.parse_transcript(bland_payload)
        # Count all turns (user + assistant) for transcript_turns
        raw_transcript = bland_payload.get("transcript", [])
        transcript_turns = len(raw_transcript) if isinstance(raw_transcript, list) else 0

        errors: list[str] = []

        # Bootstrap replay using conductor's transcript_replay method
        try:
            result = await self._conductor.transcript_replay(
                turns=user_turns,
                call_id=call_id or str(uuid.uuid4()),
                pre_classified_intents=pre_classified_intents,
            )
        except Exception as e:
            err = f"transcript_replay_failed: {e}"
            logger.error(err)
            errors.append(err)
            # Return minimal analysis rather than raising
            return CallAnalysis(
                call_id=call_id or str(uuid.uuid4()),
                bland_call_id=bland_call_id,
                final_state="greeting",
                phase="active",
                state_path=["greeting"],
                intents=[],
                turns_processed=0,
                qualification_reached=False,
                pricing_reached=False,
                offer_reached=False,
                handoff_triggered=False,
                handoff_reason=None,
                transcript_turns=transcript_turns,
                duration_seconds=duration,
                errors=errors,
            )

        state_path: list[str] = result.get("state_path", ["greeting"])
        intents: list[str] = [i for i in result.get("intents", []) if i is not None]
        final_state: str = result.get("final_state", "greeting")
        phase: str = result.get("phase", "active")

        # Check what was reached
        states_visited = set(state_path)
        qualification_reached = "qualification" in states_visited
        pricing_reached = "pricing_discussion" in states_visited
        offer_reached = bool(states_visited & _OFFER_STATES)

        # Detect handoff from phase or turn results
        handoff_triggered = phase in ("transferred",)
        handoff_reason: Optional[str] = None
        for tr in result.get("turn_results", []):
            if tr.get("handoff"):
                handoff_triggered = True
                handoff_reason = tr["handoff"].get("reason")
                break

        return CallAnalysis(
            call_id=result.get("call_id", call_id or str(uuid.uuid4())),
            bland_call_id=bland_call_id,
            final_state=final_state,
            phase=phase,
            state_path=state_path,
            intents=intents,
            turns_processed=len(user_turns),
            qualification_reached=qualification_reached,
            pricing_reached=pricing_reached,
            offer_reached=offer_reached,
            handoff_triggered=handoff_triggered,
            handoff_reason=handoff_reason,
            transcript_turns=transcript_turns,
            duration_seconds=duration,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Live-turn handler (real-time mid-call context injection)
# ---------------------------------------------------------------------------

class BlandLiveTurnHandler:
    """
    Handles Bland AI real-time tool webhook calls.

    Bland calls this endpoint mid-call via a custom tool defined in the call payload.
    We route the latest transcript turn through VoxMaestro and return context injection.

    Endpoint shape (what Bland POSTs):
        {"call_id": "...", "transcript": [...], "variables": {}, "now": "..."}

    Response shape (what Bland expects):
        {"response": "context text to inject into agent's awareness"}

    State is per-call (stored in _sessions dict keyed by call_id).
    Sessions auto-expire after 30 minutes of inactivity.
    """

    SESSION_TTL = 1800  # 30 minutes

    def __init__(self, conductor: "VoxMaestro"):
        self._conductor = conductor
        self._sessions: dict = {}  # call_id → {ctx, last_active}

    def _get_or_create_session(self, call_id: str) -> "ConversationContext":
        """Get existing session or create new one."""
        now = time.time()
        # Expire stale sessions
        expired = [k for k, v in self._sessions.items() if now - v["last_active"] > self.SESSION_TTL]
        for k in expired:
            del self._sessions[k]

        if call_id not in self._sessions:
            ctx = self._conductor.create_context(call_id=call_id)
            self._sessions[call_id] = {"ctx": ctx, "last_active": now}
        else:
            self._sessions[call_id]["last_active"] = now

        return self._sessions[call_id]["ctx"]

    def end_session(self, call_id: str) -> None:
        """Remove session (call ended)."""
        self._sessions.pop(call_id, None)

    async def handle(self, bland_payload: dict) -> dict:
        """
        Process a Bland live-turn tool webhook.
        Returns {"response": "..."} for Bland to inject.
        """
        call_id = bland_payload.get("call_id", str(uuid.uuid4()))
        transcript = bland_payload.get("transcript", [])

        # Extract only the latest user utterance
        user_turns = [t for t in transcript if isinstance(t, dict) and t.get("role") == "user"]
        if not user_turns:
            return {"response": ""}

        latest_text = user_turns[-1].get("text", "").strip()
        if not latest_text:
            return {"response": ""}

        ctx = self._get_or_create_session(call_id)

        try:
            result = await self._conductor.process_turn(ctx, latest_text)

            # Build context injection based on current state
            state = ctx.current_state
            intent = result.get("intent", "")

            # Build a concise directive for Bland's agent
            directives = []

            if result.get("handoff"):
                directives.append("TRANSFER NOW: Customer has requested to speak with a human agent.")
            elif ctx.phase.value != "active":
                directives.append(f"Call phase: {ctx.phase.value}. Wrap up gracefully.")
            else:
                # State-specific guidance
                state_guidance = {
                    "qualification": "Focus on uncovering motivation and timeline. Ask about property condition.",
                    "pricing_discussion": "Probe for asking price. Explain as-is purchase process.",
                    "offer_preparation": "Tell customer you're pulling comparable sales. Keep them engaged.",
                    "negotiation": "Empathize with price concerns. Anchor to as-is value.",
                    "closing": "Collect contact info. Confirm next steps clearly.",
                    "objection_handling": "Acknowledge concern, pivot to value proposition.",
                }
                if state in state_guidance:
                    directives.append(state_guidance[state])

                if intent == "motivated_seller":
                    directives.append("HIGH MOTIVATION DETECTED: Prioritize closing the appointment now.")
                elif intent == "needs_time":
                    directives.append("Customer needs time. Offer a specific callback time.")

            response_text = " | ".join(directives) if directives else ""

            return {
                "response": response_text,
                "_vox_meta": {  # Extra metadata (Bland ignores unknown fields)
                    "state": state,
                    "intent": intent,
                    "phase": ctx.phase.value,
                    "turn": ctx.turn_count,
                }
            }

        except Exception as e:
            logger.error("BlandLiveTurnHandler error: %s", e)
            return {"response": ""}
