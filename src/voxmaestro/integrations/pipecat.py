"""
VoxMaestro Pipecat integration.

NOTE: Pipecat is NOT currently deployed on Mac Mini.
This integration is provided for future wiring. It will not be live-tested today.
The Pipecat FrameProcessor API may differ slightly across versions — adjust process_frame
signature to match your installed pipecat version.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from ..conductor import ConversationContext, VoxMaestro

logger = logging.getLogger(__name__)

try:
    from pipecat.frames.frames import (
        Frame,
        TextFrame,
        TranscriptionFrame,
        EndFrame,
    )
    from pipecat.processors.frame_processor import FrameProcessor
    _HAS_PIPECAT = True
except ImportError:
    _HAS_PIPECAT = False
    # Stubs for type checking when pipecat not installed
    class FrameProcessor:  # type: ignore
        pass
    class Frame:  # type: ignore
        pass
    class TextFrame(Frame):  # type: ignore
        def __init__(self, text: str = ""):
            self.text = text
    class TranscriptionFrame(Frame):  # type: ignore
        def __init__(self, text: str = "", user_id: str = "", timestamp: str = ""):
            self.text = text
    class EndFrame(Frame):  # type: ignore
        pass


class VoxMaestroProcessor(FrameProcessor):
    """
    Pipecat FrameProcessor that routes transcription frames through VoxMaestro.

    Each processor instance gets its OWN ConversationContext (F2 — no callback sharing).
    Multiple processors on different calls can share one VoxMaestro instance safely.

    Usage::

        conductor = VoxMaestro.from_yaml("real_estate_agent.yaml")
        processor = VoxMaestroProcessor(conductor)
        # Wire into Pipecat pipeline: STT -> processor -> TTS
    """

    def __init__(
        self,
        conductor: VoxMaestro,
        call_id: Optional[str] = None,
        **kwargs,
    ):
        if _HAS_PIPECAT:
            super().__init__(**kwargs)
        self._conductor = conductor
        # Each processor gets its own context (F2)
        self._ctx: ConversationContext = conductor.create_context(call_id=call_id)

    @property
    def context(self) -> ConversationContext:
        return self._ctx

    async def process_frame(self, frame: Frame, direction=None) -> None:
        """Process an incoming frame."""
        if isinstance(frame, (TranscriptionFrame,)):
            text = getattr(frame, "text", "")
            if text.strip():
                try:
                    result = await self._conductor.process_turn(self._ctx, text)
                    response = result.get("response_text", "")
                    if response and _HAS_PIPECAT:
                        out_frame = TextFrame(text=response)
                        if direction is not None:
                            await self.push_frame(out_frame, direction)
                        else:
                            await self.push_frame(out_frame)
                except Exception as e:
                    logger.error("VoxMaestroProcessor error: %s", e)

        elif isinstance(frame, EndFrame):
            logger.debug("EndFrame received for call %s", self._ctx.call_id)
            if _HAS_PIPECAT:
                if direction is not None:
                    await self.push_frame(frame, direction)
                else:
                    await self.push_frame(frame)
        else:
            if _HAS_PIPECAT:
                if direction is not None:
                    await self.push_frame(frame, direction)
                else:
                    await self.push_frame(frame)
