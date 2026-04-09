"""
Tests for VoxMaestro Conductor Engine.

Run: pytest tests/ -v
"""

import asyncio
import pytest
from pathlib import Path

from voxmaestro.conductor import (
    VoxMaestro,
    ConversationContext,
    StateMachine,
    ToolBridge,
    HandoffProtocol,
    SchemaLoader,
    CallPhase,
)


EXAMPLE_CONFIG = Path(__file__).parent.parent / "examples" / "real_estate_agent.yaml"


# ─── Schema Loading ────────────────────────────────────────────────

class TestSchemaLoader:

    def test_loads_valid_config(self):
        config = SchemaLoader.load(EXAMPLE_CONFIG)
        assert config["schema_version"] == "0.1.0"
        assert config["agent"]["name"] == "dealiq-qualifier"
        assert len(config["states"]) >= 5

    def test_rejects_missing_file(self):
        with pytest.raises(FileNotFoundError):
            SchemaLoader.load("/nonexistent/path.yaml")

    def test_validates_required_keys(self):
        with pytest.raises(ValueError, match="Missing required config keys"):
            SchemaLoader._validate({"schema_version": "0.1.0"})


# ─── State Machine ─────────────────────────────────────────────────

class TestStateMachine:

    @pytest.fixture
    def config(self):
        return SchemaLoader.load(EXAMPLE_CONFIG)

    @pytest.fixture
    def sm(self, config):
        return StateMachine(config)

    @pytest.fixture
    def ctx(self):
        return ConversationContext(call_id="test-001")

    def test_initial_state_greeting_transitions_to_qualification(self, sm, ctx):
        result = sm.evaluate_transition(ctx, "greeting")
        assert result.new_state == "qualification"
        assert result.tool_to_fire is None

    def test_schedule_appointment_triggers_tool(self, sm, ctx):
        result = sm.evaluate_transition(ctx, "schedule_appointment")
        assert result.tool_to_fire == "check_availability"
        assert result.filler is not None
        assert "check" in result.filler["text"].lower()

    def test_human_request_triggers_handoff(self, sm, ctx):
        result = sm.evaluate_transition(ctx, "human_request")
        assert result.new_state == "handoff"
        assert result.trigger == "handoff"

    def test_not_interested_triggers_exit(self, sm, ctx):
        result = sm.evaluate_transition(ctx, "not_interested")
        assert result.trigger == "graceful_exit"

    def test_wildcard_transition(self, sm, ctx):
        result = sm.evaluate_transition(ctx, "unknown")
        # Wildcard "*" should send to qualification from initial
        assert result.new_state == "qualification"

    def test_max_turns_escalation(self, sm, ctx):
        ctx.current_state = "objection_handling"
        ctx.state_turn_count = 3  # max_turns is 3

        result = sm.evaluate_transition(ctx, "objection")
        assert result.new_state == "handoff"
        assert result.trigger == "max_turns_escalation"

    def test_apply_transition_updates_context(self, sm, ctx):
        from voxmaestro.conductor import TransitionResult

        result = TransitionResult(new_state="qualification")
        sm.apply_transition(ctx, result)

        assert ctx.current_state == "qualification"
        assert ctx.previous_state == "initial"
        assert ctx.state_turn_count == 0

    def test_blocked_transition_preserves_state(self, sm, ctx):
        from voxmaestro.conductor import TransitionResult

        original = ctx.current_state
        result = TransitionResult(new_state="qualification", blocked=True)
        sm.apply_transition(ctx, result)

        assert ctx.current_state == original


# ─── Tool Bridge ───────────────────────────────────────────────────

class TestToolBridge:

    @pytest.fixture
    def config(self):
        return SchemaLoader.load(EXAMPLE_CONFIG)

    @pytest.fixture
    def bridge(self, config):
        return ToolBridge(config)

    @pytest.fixture
    def ctx(self):
        return ConversationContext(call_id="test-002")

    @pytest.mark.asyncio
    async def test_filler_fires_before_tool(self, bridge, ctx):
        filler_fired = []

        async def capture_filler(filler):
            filler_fired.append(filler)

        result = await bridge.execute("check_availability", ctx, on_filler=capture_filler)

        # Filler should have fired
        assert len(filler_fired) == 1
        assert "check" in filler_fired[0]["text"].lower()

        # Tool should complete (dry run mode)
        assert result.success is True
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, bridge, ctx):
        result = await bridge.execute("nonexistent_tool", ctx)
        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_tool_result_stored_in_context(self, bridge, ctx):
        await bridge.execute("check_availability", ctx)
        assert "check_availability" in ctx.tool_results


# ─── Conductor (Integration) ──────────────────────────────────────

class TestConductor:

    @pytest.fixture
    def conductor(self):
        return VoxMaestro.from_yaml(EXAMPLE_CONFIG)

    def test_new_call_creates_context(self, conductor):
        ctx = conductor.new_call("call-001", caller_phone="+15551234567")
        assert ctx.call_id == "call-001"
        assert ctx.current_state == "initial"
        assert ctx.phase == CallPhase.ACTIVE

    @pytest.mark.asyncio
    async def test_greeting_advances_to_qualification(self, conductor):
        ctx = conductor.new_call("call-002")
        result = await conductor.process_turn(ctx, "Hi there!", intent="greeting")
        assert result["state"] == "qualification"

    @pytest.mark.asyncio
    async def test_tool_call_includes_filler(self, conductor):
        ctx = conductor.new_call("call-003")
        fillers = []

        async def capture(filler):
            fillers.append(filler)

        conductor.on_filler = capture

        result = await conductor.process_turn(
            ctx, "Do you have anything Thursday at 3?",
            intent="schedule_appointment"
        )

        assert result["tool_result"] is not None
        assert len(fillers) > 0

    @pytest.mark.asyncio
    async def test_human_request_triggers_handoff(self, conductor):
        ctx = conductor.new_call("call-004")
        result = await conductor.process_turn(
            ctx, "Can I talk to a real person?",
            intent="human_request"
        )
        assert result["action"] == "handoff"

    @pytest.mark.asyncio
    async def test_not_interested_triggers_exit(self, conductor):
        ctx = conductor.new_call("call-005")
        result = await conductor.process_turn(
            ctx, "No thanks, not interested.",
            intent="not_interested"
        )
        assert result["action"] == "exit"

    @pytest.mark.asyncio
    async def test_barge_in_returns_cancel_tts(self, conductor):
        ctx = conductor.new_call("call-006")
        result = await conductor.handle_barge_in(ctx)
        assert result["cancel_tts"] is True
        assert result["resume_stt"] is True

    @pytest.mark.asyncio
    async def test_context_tracks_turns(self, conductor):
        ctx = conductor.new_call("call-007")
        await conductor.process_turn(ctx, "Hello", intent="greeting")
        await conductor.process_turn(ctx, "Tell me about prices", intent="price_question")

        assert ctx.turn_count == 2
        assert len(ctx.intent_history) == 2
        assert ctx.intent_history[0] == "greeting"


# ─── Conversation Context ─────────────────────────────────────────

class TestConversationContext:

    def test_add_turn_increments_counters(self):
        ctx = ConversationContext(call_id="test")
        ctx.add_turn("caller", "hello", intent="greeting")

        assert ctx.turn_count == 1
        assert ctx.state_turn_count == 1
        assert len(ctx.conversation_history) == 1
        assert ctx.intent_history == ["greeting"]

    def test_duration_tracking(self):
        ctx = ConversationContext(call_id="test")
        assert ctx.duration_seconds >= 0
        assert ctx.duration_seconds < 1  # Should be nearly instant
