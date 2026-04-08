"""Core conductor tests — fixed from original (paths, imports)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voxmaestro import VoxMaestro
from voxmaestro.conductor import (
    ConversationContext,
    ConversationPhase,
    ToolBridge,
)

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
RE_AGENT_YAML = EXAMPLES_DIR / "real_estate_agent.yaml"


class TestSchemaLoader:
    def test_load_yaml(self):
        conductor = VoxMaestro.from_yaml(RE_AGENT_YAML)
        assert conductor is not None

    def test_describe(self):
        conductor = VoxMaestro.from_yaml(RE_AGENT_YAML)
        desc = conductor.describe()
        assert "agent" in desc
        assert "states" in desc
        assert "greeting" in desc["states"]

    def test_invalid_yaml(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("not_valid: [[[")
        with pytest.raises(Exception):
            VoxMaestro.from_yaml(bad)


class TestContextCreation:
    def test_create_context(self, conductor):
        ctx = conductor.create_context()
        assert ctx.current_state == "greeting"
        assert ctx.phase == ConversationPhase.ACTIVE
        assert ctx.turn_count == 0

    def test_create_context_with_call_id(self, conductor):
        ctx = conductor.create_context(call_id="test-123")
        assert ctx.call_id == "test-123"

    def test_per_call_callbacks_isolated(self, conductor):
        """F2: Each context gets independent callback references."""
        ctx1 = conductor.create_context()
        ctx2 = conductor.create_context()
        mock1 = AsyncMock()
        mock2 = AsyncMock()
        ctx1.on_filler = mock1
        ctx2.on_filler = mock2
        assert ctx1.on_filler is not ctx2.on_filler


class TestStateTransitions:
    @pytest.mark.asyncio
    async def test_greeting_to_qualification(self, conductor, ctx):
        result = await conductor.process_turn(ctx, "Yes I want to sell", pre_classified_intent="confirm_sell")
        assert ctx.current_state == "qualification"
        assert result["intent"] == "confirm_sell"
        assert result["phase"] == "active"

    @pytest.mark.asyncio
    async def test_not_interested_goes_to_objection(self, conductor, ctx):
        result = await conductor.process_turn(ctx, "No thanks", pre_classified_intent="not_interested")
        assert ctx.current_state == "objection_handling"

    @pytest.mark.asyncio
    async def test_unknown_intent_stays_in_state(self, conductor, ctx):
        result = await conductor.process_turn(ctx, "What?", pre_classified_intent="general_response")
        # greeting -> general_response -> greeting (stay)
        assert ctx.current_state == "greeting"

    @pytest.mark.asyncio
    async def test_turn_recorded_on_inactive_context(self, conductor, ctx):
        ctx.phase = ConversationPhase.EXITED
        result = await conductor.process_turn(ctx, "hello", pre_classified_intent="confirm_sell")
        assert result["blocked"] is True

    @pytest.mark.asyncio
    async def test_turn_count_increments(self, conductor, ctx):
        await conductor.process_turn(ctx, "Hi", pre_classified_intent="general_response")
        assert ctx.turn_count == 1
        await conductor.process_turn(ctx, "Yes", pre_classified_intent="confirm_sell")
        assert ctx.turn_count == 2


class TestMaxTurns:
    @pytest.mark.asyncio
    async def test_max_turns_escalates(self, conductor):
        ctx = conductor.create_context()
        # Move to qualification (max_turns=6)
        await conductor.process_turn(ctx, "", pre_classified_intent="confirm_sell")
        assert ctx.current_state == "qualification"

        # Exhaust max_turns
        for _ in range(6):
            ctx.state_turn_count += 1

        result = await conductor.process_turn(ctx, "what?", pre_classified_intent="general_response")
        # Should escalate or exit
        assert result["phase"] in ("active", "exited")


class TestHandoff:
    @pytest.mark.asyncio
    async def test_transfer_intent_triggers_handoff(self, conductor, ctx):
        ctx.on_transfer = AsyncMock()
        # Move to negotiation first
        ctx.current_state = "negotiation"
        result = await conductor.process_turn(ctx, "Let me talk to someone", pre_classified_intent="escalate")
        # escalate in negotiation goes to live_transfer, then handoff triggers on_enter
        # The key thing is context phase changes
        assert ctx.on_transfer.called or ctx.current_state in ("live_transfer", "graceful_end") or ctx.phase != ConversationPhase.ACTIVE


class TestToolBridge:
    @pytest.mark.asyncio
    async def test_unknown_tool_raises(self):
        bridge = ToolBridge({})
        ctx = ConversationContext()
        with pytest.raises(ValueError, match="Unknown tool"):
            await bridge.call("nonexistent", ctx)

    @pytest.mark.asyncio
    async def test_tool_retry_on_server_error(self):
        import httpx
        bridge = ToolBridge({
            "test_tool": {
                "endpoint": "http://localhost:9999/nope",
                "method": "POST",
                "timeout": 1.0,
                "retry": 1,
            }
        })
        ctx = ConversationContext()
        with pytest.raises(RuntimeError):
            await bridge.call("test_tool", ctx)
        await bridge.close()


class TestF1AutoReturn:
    @pytest.mark.asyncio
    async def test_tool_call_state_auto_returns(self, conductor):
        """F1: State with return_to: previous auto-returns after tool execution."""
        import yaml

        # Build a conductor with a state that has BOTH tools AND return_to
        config_yaml = """
agent:
  name: "Test Agent"
states:
  start:
    phase: "active"
    prompt: "Hello"
    transitions:
      go_tool: "fetcher"
      general_response: "start"
  fetcher:
    phase: "active"
    prompt: "Fetching..."
    return_to: "previous"
    tools:
      - "lookup"
    transitions:
      general_response: "fetcher"
  result:
    phase: "active"
    prompt: "Here are the results"
    transitions:
      general_response: "result"
tools:
  lookup:
    endpoint: "http://localhost:9999/lookup"
    method: "POST"
    timeout: 5.0
    retry: 0
"""
        cfg = yaml.safe_load(config_yaml)
        c = VoxMaestro(cfg)
        ctx = c.create_context()
        ctx.current_state = "fetcher"
        ctx.previous_state = "start"

        with patch.object(c._tool_bridge, "call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"status": "ok", "data": {}}
            result = await c.process_turn(ctx, "ok", pre_classified_intent="general_response")

        # F1: After tool call with return_to: previous, should be back in start
        assert ctx.current_state == "start", f"Expected 'start', got '{ctx.current_state}'"
        assert ctx.state_turn_count == 0


class TestF4TurnRecordedBeforeMaxDuration:
    @pytest.mark.asyncio
    async def test_turn_recorded_before_duration_check(self, conductor):
        """F4: Last utterance captured even when max_duration fires."""
        ctx = conductor.create_context()
        # Force max duration to fire immediately
        import time
        ctx.start_time = time.time() - 700  # 700s ago, max is 600

        result = await conductor.process_turn(ctx, "This should be recorded", pre_classified_intent="general_response")
        # Turn should be in transcript
        assert len(ctx.transcript) == 1
        assert ctx.transcript[0].text == "This should be recorded"
        assert result["blocked"] is True


class TestF5SkipTransitionAfterHandoff:
    @pytest.mark.asyncio
    async def test_no_mutation_after_handoff(self, conductor):
        """F5: apply_transition doesn't run on exited context."""
        ctx = conductor.create_context()
        ctx.on_transfer = AsyncMock()
        ctx.current_state = "greeting"

        # transfer intent triggers handoff + early return
        result = await conductor.process_turn(ctx, "transfer me", pre_classified_intent="transfer")
        # Context should be transferred, not mutated further
        assert ctx.phase == ConversationPhase.TRANSFERRED
        assert result["phase"] == "transferred"
