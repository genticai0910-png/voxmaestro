"""
VoxMaestro training data harvester.

Collects labeled (utterance, intent) pairs from transcript replays and saves
them as JSONL for fine-tuning dealiq-ce-v5 and vsai-intent models.

Format (one JSON object per line):
    {"text": "I need to sell fast", "intent": "motivated_seller",
     "source": "bland_replay", "call_id": "...", "state": "qualification",
     "confidence": 1.0, "timestamp": "2026-04-07T..."}

Usage:
    harvester = TrainingHarvester("~/voxmaestro/data/training/")
    harvester.record_replay(analysis, conductor)
    harvester.flush()
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conductor import VoxMaestro

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    text: str
    intent: str
    source: str          # "bland_replay" | "bland_live" | "manual"
    call_id: str
    state: str           # state when this turn occurred
    confidence: float    # 1.0 for ground-truth replays, 0.0-1.0 for live classification
    timestamp: str
    agent_name: str
    metadata: dict


class TrainingHarvester:
    """
    Collects and persists labeled training examples.
    Thread-safe via simple file append + flush pattern.
    """

    def __init__(self, data_dir: str | Path = "~/.voxmaestro/training"):
        self._dir = Path(data_dir).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[TrainingExample] = []
        self._current_file = self._dir / f"examples_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"

    def record(self, example: TrainingExample) -> None:
        """Add a single example to the buffer."""
        self._buffer.append(example)

    def record_turn(
        self,
        text: str,
        intent: str,
        state: str,
        call_id: str,
        source: str = "bland_replay",
        confidence: float = 1.0,
        agent_name: str = "unknown",
        **metadata,
    ) -> None:
        """Convenience method to record a single turn."""
        self.record(TrainingExample(
            text=text,
            intent=intent,
            source=source,
            call_id=call_id,
            state=state,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=agent_name,
            metadata=metadata,
        ))

    def record_replay(self, analysis: "CallAnalysis", conductor: "VoxMaestro") -> int:
        """
        Extract all labeled turns from a CallAnalysis and add to buffer.
        Returns number of examples added.
        """
        from .integrations.bland import CallAnalysis  # type: ignore

        agent_name = conductor._agent.get("name", "unknown")
        count = 0

        # Walk transcript — match user turns to intents
        # analysis has intents list aligned with state_path
        # pair them up: intents[i] corresponds to turn i of the transcript

        # Build from the parallel lists in analysis
        for i, intent in enumerate(analysis.intents):
            if not intent:
                continue
            # state_path[i] is state before turn i; state_path[i+1] after
            state = analysis.state_path[i] if i < len(analysis.state_path) else "unknown"

            # We don't have the raw text here — use metadata if available
            turn_text = analysis.metadata.get(f"turn_{i}_text", "")
            if not turn_text:
                continue  # skip turns without text (can't use for training)

            self.record_turn(
                text=turn_text,
                intent=intent,
                state=state,
                call_id=analysis.call_id,
                source="bland_replay",
                confidence=1.0,  # replay = ground truth
                agent_name=agent_name,
            )
            count += 1

        return count

    def record_live_turn(
        self,
        text: str,
        intent: str,
        state: str,
        call_id: str,
        confidence: float = 0.8,
        agent_name: str = "unknown",
    ) -> None:
        """Record a live-classified turn (from BlandLiveTurnHandler)."""
        self.record_turn(
            text=text,
            intent=intent,
            state=state,
            call_id=call_id,
            source="bland_live",
            confidence=confidence,
            agent_name=agent_name,
        )

    def flush(self) -> int:
        """Write buffered examples to JSONL file. Returns count written."""
        if not self._buffer:
            return 0
        count = 0
        with open(self._current_file, "a", encoding="utf-8") as f:
            for ex in self._buffer:
                f.write(json.dumps({
                    "text": ex.text,
                    "intent": ex.intent,
                    "source": ex.source,
                    "call_id": ex.call_id,
                    "state": ex.state,
                    "confidence": ex.confidence,
                    "timestamp": ex.timestamp,
                    "agent_name": ex.agent_name,
                    **ex.metadata,
                }, ensure_ascii=False) + "\n")
                count += 1
        self._buffer.clear()
        logger.info("training_harvester_flush count=%d file=%s", count, str(self._current_file))
        return count

    def stats(self) -> dict:
        """Return stats about saved training data."""
        files = sorted(self._dir.glob("*.jsonl"))
        total = 0
        intent_counts: dict[str, int] = {}
        for f in files:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    total += 1
                    intent = obj.get("intent", "unknown")
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
                except json.JSONDecodeError:
                    pass
        return {
            "total_examples": total,
            "files": len(files),
            "intent_distribution": intent_counts,
            "data_dir": str(self._dir),
        }

    def export_alpaca(self, output_path: str | Path) -> int:
        """
        Export all examples in Alpaca fine-tuning format (instruction/input/output).
        Compatible with the existing DealiQ fine-tuning pipeline.

        Format:
            {"instruction": "Classify...", "input": "<utterance>", "output": "<intent>"}
        """
        output = Path(output_path)
        files = sorted(self._dir.glob("*.jsonl"))
        count = 0
        with open(output, "w", encoding="utf-8") as out:
            for f in files:
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        out.write(json.dumps({
                            "instruction": (
                                "You are a real estate voice agent intent classifier. "
                                "Classify the caller utterance into the correct intent label."
                            ),
                            "input": obj["text"],
                            "output": obj["intent"],
                            "metadata": {
                                "state": obj.get("state"),
                                "source": obj.get("source"),
                                "confidence": obj.get("confidence"),
                                "agent_name": obj.get("agent_name"),
                            }
                        }, ensure_ascii=False) + "\n")
                        count += 1
                    except (json.JSONDecodeError, KeyError):
                        pass
        logger.info("alpaca_export count=%d output=%s", count, str(output))
        return count
