"""
VoxMaestro iRELOP integration.

Maps voice conversation analysis (CallAnalysis) to iRELOP scoring signals.
The voice layer adds a Motivation bonus and timeline hints derived from the
conversation flow — layered on top of property data from prop-enricher.

Scoring contribution:
  Motivation (voice):   up to +20pts based on conversation depth + urgency intents
  Opportunity (voice):  not directly — equity comes from prop data, not voice
  Timeline hint:        extracted intent flags → timeline_days estimate

Usage:
    from voxmaestro.integrations.irelop import VoxIRELOPEnricher
    enricher = VoxIRELOPEnricher()
    signals = enricher.extract_signals(analysis)
    # signals is a dict ready to merge into iRELOP lead_data
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .bland import CallAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VoiceSignals dataclass
# ---------------------------------------------------------------------------

@dataclass
class VoiceSignals:
    """Voice-derived signals for iRELOP scoring."""
    motivation_bonus: int           # 0-20 pts to add to Motivation component
    timeline_days: Optional[int]    # estimated urgency in days (30/90/180/None)
    distress_signals: list          # voice-detected signals: "motivated_seller", etc.
    condition_hint: Optional[str]   # "distressed"|"fair"|"average"|None
    is_decision_maker: bool         # True if caller is the owner (assumed True from voice)
    voice_score: int                # 0-100 qualification depth (from qualification_score)
    voice_tier: str                 # HOT/WARM/COOL/PASS based on voice_score alone
    call_duration_seconds: Optional[float]
    state_path: list
    raw_intents: list


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------

class VoxIRELOPEnricher:
    """Converts CallAnalysis into iRELOP-compatible lead scoring signals."""

    # intent → distress signal mapping
    INTENT_DISTRESS_MAP = {
        "motivated_seller": "motivated_seller",
        "urgent_timeline": "urgent_timeline",
        "financial_pressure": "financial_pressure",
        "divorce": "divorce",
        "foreclosure": "foreclosure",
        "probate": "probate",
        "vacant_property": "vacant",
        "needs_repairs": "needs_work",
    }

    # intent → timeline_days estimate
    INTENT_TIMELINE_MAP = {
        "motivated_seller": 30,
        "urgent_timeline": 14,
        "needs_time": 180,
        "follow_up_scheduled": 90,
        "timeline_question": 60,
    }

    # voice_score → motivation_bonus (ordered high-to-low, first match wins)
    MOTIVATION_BONUS_MAP = [
        (90, 20),   # score >= 90 → +20
        (75, 15),   # score >= 75 → +15
        (50, 10),   # score >= 50 → +10
        (25, 5),    # score >= 25 → +5
        (0,  0),    # score < 25  → +0
    ]

    def extract_signals(self, analysis: "CallAnalysis") -> VoiceSignals:
        """Extract iRELOP-compatible signals from a CallAnalysis."""
        from .bland import qualification_score

        score = qualification_score(analysis)

        # Motivation bonus from voice depth
        motivation_bonus = next(
            (bonus for threshold, bonus in self.MOTIVATION_BONUS_MAP if score >= threshold), 0
        )

        # Distress signals from intents
        distress = [
            self.INTENT_DISTRESS_MAP[intent]
            for intent in analysis.intents
            if intent in self.INTENT_DISTRESS_MAP
        ]
        # Also from state path
        if "objection_handling" in analysis.state_path and analysis.offer_reached:
            distress.append("motivated_despite_objections")

        # Timeline from intents — take most urgent (smallest value)
        timeline: Optional[int] = None
        for intent in analysis.intents:
            if intent in self.INTENT_TIMELINE_MAP:
                t = self.INTENT_TIMELINE_MAP[intent]
                if timeline is None or t < timeline:
                    timeline = t

        # Condition hint from intents
        condition: Optional[str] = None
        if "needs_repairs" in analysis.intents or "distressed_property" in analysis.intents:
            condition = "distressed"

        # Voice tier (based solely on voice score)
        if score >= 80:
            voice_tier = "HOT"
        elif score >= 60:
            voice_tier = "WARM"
        elif score >= 40:
            voice_tier = "COOL"
        else:
            voice_tier = "PASS"

        return VoiceSignals(
            motivation_bonus=motivation_bonus,
            timeline_days=timeline,
            distress_signals=list(set(distress)),
            condition_hint=condition,
            is_decision_maker=True,  # voice calls are always decision makers
            voice_score=score,
            voice_tier=voice_tier,
            call_duration_seconds=analysis.duration_seconds,
            state_path=list(analysis.state_path),
            raw_intents=list(analysis.intents),
        )

    def to_lead_data_patch(
        self,
        signals: VoiceSignals,
        existing_lead_data: Optional[dict] = None,
    ) -> dict:
        """
        Convert VoiceSignals to a lead_data dict patch suitable for iRELOP scoring.
        Merges with existing_lead_data if provided (voice signals take priority for
        motivation fields, existing data wins for property fields).
        """
        base = dict(existing_lead_data or {})

        # Merge distress signals (additive)
        existing_distress = base.get("distress_signals", [])
        merged_distress = list(set(existing_distress + signals.distress_signals))
        base["distress_signals"] = merged_distress

        # Timeline: voice wins if more urgent
        if signals.timeline_days is not None:
            existing_timeline = base.get("timeline_days", 999)
            base["timeline_days"] = min(signals.timeline_days, existing_timeline)

        # Condition hint: only set if not already known
        if signals.condition_hint and not base.get("condition"):
            base["condition"] = signals.condition_hint

        # Decision maker: voice = always true
        base["is_decision_maker"] = True

        # Voice metadata
        base["voice_score"] = signals.voice_score
        base["voice_tier"] = signals.voice_tier
        base["motivation_bonus"] = signals.motivation_bonus

        return base

    async def enrich_and_post(
        self,
        analysis: "CallAnalysis",
        lead_id: str,
        webhook_url: str,
        existing_lead_data: Optional[dict] = None,
    ) -> dict:
        """
        Extract signals, build lead_data patch, and POST to the iRELOP webhook.

        Args:
            analysis: CallAnalysis from VoxMaestro replay.
            lead_id: The iRELOP lead identifier.
            webhook_url: n8n /webhook/lead-enrich endpoint URL.
            existing_lead_data: Optional property data to merge with voice signals.

        Returns:
            {"status": "ok", "signals": VoiceSignals, "response": <webhook response>}
            or {"status": "error", "error": "...", "signals": VoiceSignals}
        """
        import httpx
        from .bland import qualification_score

        signals = self.extract_signals(analysis)
        lead_data = self.to_lead_data_patch(signals, existing_lead_data)

        payload = {
            "lead_id": lead_id,
            "source": "voxmaestro_voice",
            "voice_signals": asdict(signals),
            "lead_data": lead_data,
            "call_analysis": {
                "final_state": analysis.final_state,
                "score": qualification_score(analysis),
                "state_path": analysis.state_path,
                "turns_processed": analysis.turns_processed,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(webhook_url, json=payload)
                resp.raise_for_status()
                webhook_response = resp.json()
            logger.info(
                "irelop_enrich_posted",
                lead_id=lead_id,
                voice_score=signals.voice_score,
                voice_tier=signals.voice_tier,
            )
            return {"status": "ok", "signals": signals, "response": webhook_response}
        except Exception as e:
            logger.error("irelop_enrich_post_failed", lead_id=lead_id, error=str(e))
            return {"status": "error", "error": str(e), "signals": signals}
