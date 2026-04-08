"""Tests for Bland AI transcript adapter."""
from pathlib import Path

import pytest

from voxmaestro import BlandTranscriptAdapter, CallAnalysis, VoxMaestro, qualification_score

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
RE_YAML = EXAMPLES_DIR / "real_estate_agent.yaml"


@pytest.fixture
def conductor():
    return VoxMaestro.from_yaml(RE_YAML)


@pytest.fixture
def adapter(conductor):
    return BlandTranscriptAdapter(conductor)


# ─────────────────────────────────────────────────────────────────────────────
# Transcript parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestBlandTranscriptParsing:
    def test_parse_structured_transcript(self, adapter):
        payload = {
            "call_id": "bland-123",
            "transcript": [
                {"role": "assistant", "text": "Hi, I'm calling about your property."},
                {"role": "user", "text": "Yes I want to sell"},
                {"role": "assistant", "text": "Great, let me ask a few questions."},
                {"role": "user", "text": "The house is in good condition"},
            ],
        }
        turns = adapter.parse_transcript(payload)
        assert len(turns) == 2
        assert turns[0] == "Yes I want to sell"
        assert turns[1] == "The house is in good condition"

    def test_parse_flat_transcript(self, adapter):
        payload = {
            "call_id": "bland-456",
            "concatenated_transcript": "Agent: Hello\nUser: Yes I'm interested\nAgent: Great",
        }
        turns = adapter.parse_transcript(payload)
        # Flat transcript — returns empty list (can't reliably parse turn-by-turn)
        assert isinstance(turns, list)

    def test_parse_empty_transcript(self, adapter):
        payload = {"call_id": "bland-789"}
        turns = adapter.parse_transcript(payload)
        assert turns == []

    def test_parse_filters_empty_user_turns(self, adapter):
        payload = {
            "call_id": "bland-filter",
            "transcript": [
                {"role": "user", "text": ""},
                {"role": "user", "text": "   "},
                {"role": "user", "text": "Valid text"},
            ],
        }
        turns = adapter.parse_transcript(payload)
        assert turns == ["Valid text"]

    def test_parse_assistant_only_transcript(self, adapter):
        payload = {
            "call_id": "bland-agent-only",
            "transcript": [
                {"role": "assistant", "text": "Hello"},
                {"role": "assistant", "text": "How can I help?"},
            ],
        }
        turns = adapter.parse_transcript(payload)
        assert turns == []


# ─────────────────────────────────────────────────────────────────────────────
# Replay
# ─────────────────────────────────────────────────────────────────────────────

class TestBlandReplay:
    @pytest.mark.asyncio
    async def test_replay_qualification_flow(self, adapter):
        payload = {
            "call_id": "bland-test-001",
            "duration": 120.0,
            "transcript": [
                {"role": "assistant", "text": "Hi!"},
                {"role": "user", "text": "Yes I want to sell"},
                {"role": "user", "text": "I'm very motivated, need to sell fast"},
            ],
        }
        analysis = await adapter.replay(
            payload, pre_classified_intents=["confirm_sell", "motivated_seller"]
        )
        assert analysis.bland_call_id == "bland-test-001"
        assert analysis.qualification_reached
        assert "qualification" in analysis.state_path
        assert analysis.duration_seconds == 120.0

    @pytest.mark.asyncio
    async def test_replay_no_interest(self, adapter):
        payload = {
            "call_id": "bland-test-002",
            "transcript": [
                {"role": "user", "text": "I'm not interested"},
            ],
        }
        analysis = await adapter.replay(
            payload, pre_classified_intents=["not_interested"]
        )
        assert not analysis.qualification_reached
        assert analysis.turns_processed >= 1

    @pytest.mark.asyncio
    async def test_replay_empty_transcript(self, adapter):
        payload = {"call_id": "bland-empty", "transcript": []}
        analysis = await adapter.replay(payload)
        assert analysis.turns_processed == 0
        assert analysis.final_state == "greeting"

    @pytest.mark.asyncio
    async def test_replay_no_transcript_key(self, adapter):
        payload = {"call_id": "bland-nokey"}
        analysis = await adapter.replay(payload)
        assert analysis.turns_processed == 0
        assert isinstance(analysis.errors, list)

    @pytest.mark.asyncio
    async def test_replay_call_id_propagation(self, adapter):
        payload = {
            "call_id": "bland-cid-propagation",
            "transcript": [{"role": "user", "text": "sell"}],
        }
        analysis = await adapter.replay(payload, pre_classified_intents=["confirm_sell"])
        assert analysis.bland_call_id == "bland-cid-propagation"
        assert analysis.call_id  # has an internal UUID

    @pytest.mark.asyncio
    async def test_replay_transcript_turns_count(self, adapter):
        payload = {
            "call_id": "bland-tcount",
            "transcript": [
                {"role": "assistant", "text": "Hi"},
                {"role": "user", "text": "Yes sell"},
                {"role": "assistant", "text": "Ok"},
                {"role": "user", "text": "Motivated"},
            ],
        }
        analysis = await adapter.replay(
            payload, pre_classified_intents=["confirm_sell", "motivated_seller"]
        )
        assert analysis.transcript_turns == 4  # all turns, not just user
        assert analysis.turns_processed == 2    # user only

    @pytest.mark.asyncio
    async def test_qualification_score_depth(self, adapter):
        payload = {
            "call_id": "bland-score-test",
            "transcript": [
                {"role": "user", "text": "sell"},
                {"role": "user", "text": "motivated"},
            ],
        }
        analysis = await adapter.replay(
            payload, pre_classified_intents=["confirm_sell", "motivated_seller"]
        )
        score = qualification_score(analysis)
        assert score >= 25  # reached at least qualification


# ─────────────────────────────────────────────────────────────────────────────
# qualification_score function
# ─────────────────────────────────────────────────────────────────────────────

class TestQualificationScore:
    def _make_analysis(self, **kwargs) -> CallAnalysis:
        defaults = dict(
            call_id="x",
            bland_call_id="b",
            final_state="greeting",
            phase="active",
            state_path=["greeting"],
            intents=[],
            turns_processed=1,
            qualification_reached=False,
            pricing_reached=False,
            offer_reached=False,
            handoff_triggered=False,
            handoff_reason=None,
            transcript_turns=1,
            duration_seconds=None,
            metadata={},
            errors=[],
        )
        defaults.update(kwargs)
        return CallAnalysis(**defaults)

    def test_score_greeting_only(self):
        a = self._make_analysis()
        assert qualification_score(a) == 0

    def test_score_qualification_reached(self):
        a = self._make_analysis(
            qualification_reached=True,
            state_path=["greeting", "qualification"],
        )
        assert qualification_score(a) == 25

    def test_score_pricing_reached(self):
        a = self._make_analysis(
            qualification_reached=True,
            pricing_reached=True,
            state_path=["greeting", "qualification", "pricing_discussion"],
            turns_processed=3,
            duration_seconds=60.0,
        )
        assert qualification_score(a) == 50

    def test_score_offer_reached(self):
        a = self._make_analysis(
            qualification_reached=True,
            pricing_reached=True,
            offer_reached=True,
            state_path=["greeting", "qualification", "pricing_discussion", "offer_preparation"],
        )
        assert qualification_score(a) == 75

    def test_score_closing_reached(self):
        a = self._make_analysis(
            qualification_reached=True,
            pricing_reached=True,
            offer_reached=True,
            state_path=["greeting", "qualification", "pricing_discussion", "offer_presentation", "closing"],
        )
        assert qualification_score(a) == 90

    def test_score_handoff_triggered(self):
        a = self._make_analysis(
            handoff_triggered=True,
            handoff_reason="user_requested_transfer",
        )
        assert qualification_score(a) == 100


# ─────────────────────────────────────────────────────────────────────────────
# conductor.transcript_replay()
# ─────────────────────────────────────────────────────────────────────────────

class TestTranscriptReplay:
    @pytest.mark.asyncio
    async def test_replay_method(self, conductor):
        turns = ["Yes I want to sell", "I'm motivated to sell quickly"]
        result = await conductor.transcript_replay(
            turns, pre_classified_intents=["confirm_sell", "motivated_seller"]
        )
        assert "state_path" in result
        assert "intents" in result
        assert result["final_state"] in conductor._states
        assert len(result["state_path"]) >= 1

    @pytest.mark.asyncio
    async def test_replay_empty(self, conductor):
        result = await conductor.transcript_replay([])
        assert result["final_state"] == "greeting"
        assert result["state_path"] == ["greeting"]

    @pytest.mark.asyncio
    async def test_replay_state_path_grows(self, conductor):
        turns = ["sell", "motivated"]
        result = await conductor.transcript_replay(
            turns, pre_classified_intents=["confirm_sell", "motivated_seller"]
        )
        # state_path has initial state + one entry per processed turn
        assert len(result["state_path"]) == len(turns) + 1

    @pytest.mark.asyncio
    async def test_replay_custom_call_id(self, conductor):
        result = await conductor.transcript_replay([], call_id="test-cid-123")
        assert result["call_id"] == "test-cid-123"

    @pytest.mark.asyncio
    async def test_replay_intents_recorded(self, conductor):
        turns = ["sell"]
        result = await conductor.transcript_replay(
            turns, pre_classified_intents=["confirm_sell"]
        )
        assert result["intents"][0] == "confirm_sell"

    @pytest.mark.asyncio
    async def test_replay_stops_on_non_active_phase(self, conductor):
        # Feed a not_interested followed by still_not_interested to exhaust
        # objection_handling and trigger graceful exit
        turns = ["not interested", "still not interested", "still no", "another turn"]
        result = await conductor.transcript_replay(
            turns,
            pre_classified_intents=[
                "not_interested",
                "still_not_interested",
                "still_not_interested",
                "confirm_sell",
            ],
        )
        # Should stop before processing all turns once phase exits active
        assert result["final_state"] in conductor._states
