"""
VoxMaestro ↔ Pipecat Integration

Bridges VoxMaestro's state machine and tool bridge into Pipecat's
frame-based audio pipeline. This is the glue between the conductor
(conversation logic) and the transport (real-time audio).

Usage:
    from voxmaestro.conductor import VoxMaestro
    from voxmaestro.integrations.pipecat import VoxMaestroPipecatProcessor

    conductor = VoxMaestro.from_yaml("agent.yaml")

    processor = VoxMaestroPipecatProcessor(
        conductor=conductor,
        intent_classifier=my_vsai_classifier,
    )

    pipeline = Pipeline([
        transport.input(),
        stt_processor,
        processor,          # ← VoxMaestro sits between STT and LLM
        llm_processor,
        tts_processor,
        transport.output(),
    ])
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from voxmaestro.conductor import CallPhase, ConversationContext, VoxMaestro

logger = logging.getLogger("voxmaestro.pipecat")


# ─── Frame Types (Pipecat-compatible) ───────────────────────────────
# These mirror Pipecat's frame hierarchy for zero-friction integration.

class VoxFrame:
    """Base frame type for VoxMaestro pipeline events."""
    pass


class FillerFrame(VoxFrame):
    """
    Injected into the TTS output IMMEDIATELY when a tool call fires.
    Bypasses the LLM entirely — this is the pre-LLM gate.
    """
    def __init__(self, text: str, audio_path: Optional[str] = None):
        self.text = text
        self.audio_path = audio_path  # Pre-rendered WAV for zero latency


class ToolResultFrame(VoxFrame):
    """Carries tool call results back into the LLM context."""
    def __init__(self, tool_name: str, result: Any, success: bool):
        self.tool_name = tool_name
        self.result = result
        self.success = success


class StateChangeFrame(VoxFrame):
    """Notifies the pipeline of a state machine transition."""
    def __init__(self, from_state: str, to_state: str, trigger: Optional[str] = None):
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger


class HandoffFrame(VoxFrame):
    """Signals the pipeline to initiate telephony transfer."""
    def __init__(self, payload: dict):
        self.payload = payload


class BargeInFrame(VoxFrame):
    """Signals the pipeline to cancel TTS and resume STT."""
    def __init__(self):
        self.cancel_tts = True
        self.resume_stt = True
        self.flush_buffer = True


# ─── Pipecat Frame Processor ───────────────────────────────────────

class VoxMaestroPipecatProcessor:
    """
    Pipecat frame processor that integrates VoxMaestro's conductor.

    Sits between STT and LLM in the Pipecat pipeline:

        STT → [VoxMaestro] → LLM → TTS

    On each transcribed utterance:
      1. Classifies intent (via pluggable classifier)
      2. Evaluates state transition
      3. If tool needed: emits FillerFrame → fires tool → emits ToolResultFrame
      4. If handoff: emits HandoffFrame
      5. Otherwise: passes through to LLM for generation
    """

    def __init__(
        self,
        conductor: VoxMaestro,
        intent_classifier: Optional[Callable] = None,
        context: Optional[ConversationContext] = None,
    ):
        self.conductor = conductor
        self.intent_classifier = intent_classifier
        self._ctx = context
        self._output_queue: asyncio.Queue[VoxFrame] = asyncio.Queue()

    @property
    def ctx(self) -> ConversationContext:
        if self._ctx is None:
            raise RuntimeError("No active call context. Call start_call() first.")
        return self._ctx

    def start_call(self, call_id: str, caller_phone: str = "", **kwargs) -> ConversationContext:
        """Initialize a new call context."""
        self._ctx = self.conductor.new_call(call_id, caller_phone, **kwargs)

        # Wire conductor callbacks to frame emission
        self.conductor.on_filler = self._emit_filler
        self.conductor.on_transfer = self._emit_handoff

        return self._ctx

    async def process_frame(self, frame: Any) -> list[VoxFrame]:
        """
        Process an incoming frame from the Pipecat pipeline.

        In Pipecat, this would be called by the pipeline runner
        whenever an upstream processor (STT) emits a frame.

        For a TranscriptionFrame (STT output), we:
          1. Classify the intent
          2. Run the conductor
          3. Emit appropriate downstream frames
        """

        # Check if this is a transcription frame (text from STT)
        text = getattr(frame, "text", None)
        if not text:
            return []

        output_frames = []

        # Classify intent
        intent = None
        if self.intent_classifier:
            intent = await self.intent_classifier(text, self.ctx)

        # Run conductor
        result = await self.conductor.process_turn(self.ctx, text, intent=intent)

        # Emit state change frame
        if result["state"] != self.ctx.previous_state:
            output_frames.append(
                StateChangeFrame(
                    from_state=self.ctx.previous_state,
                    to_state=result["state"],
                    trigger=result.get("action"),
                )
            )

        # Handle filler (already emitted via callback, but track it)
        if result.get("filler"):
            filler = result["filler"]
            output_frames.append(
                FillerFrame(
                    text=filler.get("text", ""),
                    audio_path=filler.get("audio"),
                )
            )

        # Handle tool result
        if result.get("tool_result"):
            tr = result["tool_result"]
            output_frames.append(
                ToolResultFrame(
                    tool_name=tr.tool_name,
                    result=tr.data,
                    success=tr.success,
                )
            )

        # Handle handoff
        if result.get("action") == "handoff":
            output_frames.append(HandoffFrame(payload=result.get("handoff", {})))

        # Handle exit
        if result.get("action") == "exit":
            # Response text goes to TTS, then pipeline closes
            pass

        # Drain any frames from callback queue
        while not self._output_queue.empty():
            output_frames.append(self._output_queue.get_nowait())

        return output_frames

    async def handle_barge_in(self) -> BargeInFrame:
        """Called by the audio transport when barge-in is detected."""
        await self.conductor.handle_barge_in(self.ctx)
        return BargeInFrame()

    async def handle_silence(self) -> Optional[FillerFrame]:
        """Called when silence timeout is reached."""
        result = await self.conductor.handle_silence(self.ctx)
        if result.get("response_text"):
            return FillerFrame(text=result["response_text"])
        return None

    # ── Internal Callbacks ──

    async def _emit_filler(self, filler: dict):
        """Callback: conductor wants to play a filler."""
        frame = FillerFrame(
            text=filler.get("text", ""),
            audio_path=filler.get("audio"),
        )
        await self._output_queue.put(frame)

    async def _emit_handoff(self, teardown_data: dict):
        """Callback: conductor is tearing down for handoff."""
        frame = HandoffFrame(payload=teardown_data)
        await self._output_queue.put(frame)
