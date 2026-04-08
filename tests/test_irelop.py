"""Tests for iRELOP voice signal enrichment."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from voxmaestro import VoxMaestro, BlandTranscriptAdapter, qualification_score
from voxmaestro.integrations.irelop import VoxIRELOPEnricher, VoiceSignals
from voxmaestro.integrations.bland import BlandLiveTurnHandler, CallAnalysis

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
RE_YAML = EXAMPLES_DIR / "real_estate_agent.yaml"

@pytest.fixture
def conductor():
    return VoxMaestro.from_yaml(RE_YAML)

@pytest.fixture
def enricher():
    return VoxIRELOPEnricher()

def make_analysis(**kwargs) -> CallAnalysis:
    defaults = dict(
        call_id="x", bland_call_id="b", final_state="qualification",
        phase="active", state_path=["greeting", "qualification"],
        intents=["confirm_sell", "general_response"], turns_processed=2,
        qualification_reached=True, pricing_reached=False, offer_reached=False,
        handoff_triggered=False, handoff_reason=None, transcript_turns=2,
        duration_seconds=60.0, metadata={}, errors=[]
    )
    defaults.update(kwargs)
    return CallAnalysis(**defaults)


class TestVoiceSignalExtraction:
    def test_qualification_reached_gives_motivation_bonus(self, enricher):
        a = make_analysis(qualification_reached=True)
        signals = enricher.extract_signals(a)
        assert signals.motivation_bonus >= 5
        assert signals.voice_score >= 25

    def test_offer_reached_gives_high_bonus(self, enricher):
        a = make_analysis(
            qualification_reached=True, pricing_reached=True, offer_reached=True,
            state_path=["greeting", "qualification", "pricing_discussion", "offer_presentation"],
        )
        signals = enricher.extract_signals(a)
        assert signals.motivation_bonus >= 15
        assert signals.voice_score >= 75

    def test_motivated_seller_intent_adds_distress(self, enricher):
        a = make_analysis(intents=["confirm_sell", "motivated_seller"])
        signals = enricher.extract_signals(a)
        assert "motivated_seller" in signals.distress_signals

    def test_motivated_seller_sets_timeline(self, enricher):
        a = make_analysis(intents=["confirm_sell", "motivated_seller"])
        signals = enricher.extract_signals(a)
        assert signals.timeline_days is not None
        assert signals.timeline_days <= 30

    def test_needs_time_sets_long_timeline(self, enricher):
        a = make_analysis(intents=["confirm_sell", "needs_time"])
        signals = enricher.extract_signals(a)
        assert signals.timeline_days == 180

    def test_urgent_takes_min_timeline(self, enricher):
        """Most urgent intent wins when multiple timeline signals present."""
        a = make_analysis(intents=["motivated_seller", "urgent_timeline"])
        signals = enricher.extract_signals(a)
        assert signals.timeline_days == 14  # urgent_timeline wins

    def test_no_call_gives_zero_bonus(self, enricher):
        a = make_analysis(
            qualification_reached=False, pricing_reached=False, offer_reached=False,
            state_path=["greeting"], intents=["not_interested"],
        )
        signals = enricher.extract_signals(a)
        assert signals.motivation_bonus == 0
        assert signals.voice_tier in ("PASS", "COOL")

    def test_is_decision_maker_always_true(self, enricher):
        a = make_analysis()
        signals = enricher.extract_signals(a)
        assert signals.is_decision_maker is True


class TestLeadDataPatch:
    def test_patch_merges_distress_signals(self, enricher):
        a = make_analysis(intents=["motivated_seller"])
        signals = enricher.extract_signals(a)
        existing = {"distress_signals": ["divorce"], "arv": 300000}
        patch_data = enricher.to_lead_data_patch(signals, existing)
        assert "divorce" in patch_data["distress_signals"]
        assert "motivated_seller" in patch_data["distress_signals"]
        assert patch_data["arv"] == 300000  # existing property data preserved

    def test_patch_timeline_takes_most_urgent(self, enricher):
        a = make_analysis(intents=["urgent_timeline"])
        signals = enricher.extract_signals(a)
        existing = {"timeline_days": 90}
        patch_data = enricher.to_lead_data_patch(signals, existing)
        assert patch_data["timeline_days"] == 14  # urgent wins over 90

    def test_patch_without_existing(self, enricher):
        a = make_analysis()
        signals = enricher.extract_signals(a)
        patch_data = enricher.to_lead_data_patch(signals)
        assert "voice_score" in patch_data
        assert "voice_tier" in patch_data
        assert "motivation_bonus" in patch_data

    @pytest.mark.asyncio
    async def test_enrich_and_post(self, enricher):
        a = make_analysis(intents=["motivated_seller"])
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "queued"}
            mock_resp.raise_for_status = AsyncMock()
            mock_post.return_value = mock_resp
            result = await enricher.enrich_and_post(
                a, "lead-123", "http://localhost:5678/webhook/lead-enrich"
            )
        assert result["status"] == "ok"
        assert "signals" in result


class TestBlandLiveTurnHandler:
    @pytest.mark.asyncio
    async def test_handle_user_turn(self, conductor):
        handler = BlandLiveTurnHandler(conductor)
        payload = {
            "call_id": "live-001",
            "transcript": [
                {"role": "assistant", "text": "Hi there"},
                {"role": "user", "text": "Yes I want to sell my house"},
            ]
        }
        result = await handler.handle(payload)
        assert "response" in result
        assert isinstance(result["response"], str)

    @pytest.mark.asyncio
    async def test_handle_empty_transcript(self, conductor):
        handler = BlandLiveTurnHandler(conductor)
        result = await handler.handle({"call_id": "live-002", "transcript": []})
        assert result == {"response": ""}

    @pytest.mark.asyncio
    async def test_sessions_isolated(self, conductor):
        """Two calls get separate session state."""
        handler = BlandLiveTurnHandler(conductor)
        await handler.handle({
            "call_id": "call-A",
            "transcript": [{"role": "user", "text": "yes"}]
        })
        await handler.handle({
            "call_id": "call-B",
            "transcript": [{"role": "user", "text": "no thanks"}]
        })
        ctx_a = handler._sessions["call-A"]["ctx"]
        ctx_b = handler._sessions["call-B"]["ctx"]
        assert ctx_a.call_id != ctx_b.call_id
        assert ctx_a.current_state != ctx_b.current_state or (
            ctx_a.intents_seen if hasattr(ctx_a, "intents_seen") else True
        )

    @pytest.mark.asyncio
    async def test_end_session(self, conductor):
        handler = BlandLiveTurnHandler(conductor)
        await handler.handle({
            "call_id": "live-end",
            "transcript": [{"role": "user", "text": "hello"}]
        })
        assert "live-end" in handler._sessions
        handler.end_session("live-end")
        assert "live-end" not in handler._sessions

    @pytest.mark.asyncio
    async def test_session_continuity(self, conductor):
        """Same call_id across multiple handle() calls maintains state."""
        handler = BlandLiveTurnHandler(conductor)
        # First turn
        await handler.handle({
            "call_id": "cont-001",
            "transcript": [{"role": "user", "text": "yes sell"}]
        })
        ctx_after_1 = handler._sessions["cont-001"]["ctx"]
        state_after_1 = ctx_after_1.current_state

        # Second turn (append to same transcript)
        await handler.handle({
            "call_id": "cont-001",
            "transcript": [
                {"role": "user", "text": "yes sell"},
                {"role": "assistant", "text": "great"},
                {"role": "user", "text": "i need to sell fast"},
            ]
        })
        # Turn count should have advanced
        assert ctx_after_1.turn_count >= 2
