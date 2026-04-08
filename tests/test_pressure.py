"""
VoxMaestro pressure tests — adversarial, concurrency, and edge case scenarios.
"""

import asyncio
import random
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from voxmaestro import VoxMaestro
from voxmaestro.conductor import ConversationContext, ConversationPhase, ToolBridge

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
RE_AGENT_YAML = EXAMPLES_DIR / "real_estate_agent.yaml"


@pytest.fixture
def conductor() -> VoxMaestro:
    return VoxMaestro.from_yaml(RE_AGENT_YAML)


# ---------------------------------------------------------------------------
# P1: Concurrent calls — shared conductor, no cross-contamination
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p1_ten_concurrent_calls(conductor):
    """10 parallel calls on shared conductor must not cross-contaminate state."""
    async def run_call(call_id: str) -> dict:
        ctx = conductor.create_context(call_id=call_id)
        results = []
        intents = ["confirm_sell", "general_response", "not_interested"]
        for i in range(3):
            r = await conductor.process_turn(ctx, f"turn {i}", pre_classified_intent=intents[i % len(intents)])
            results.append((ctx.call_id, ctx.current_state, ctx.turn_count))
        return {"call_id": call_id, "final_state": ctx.current_state, "turns": ctx.turn_count}

    tasks = [run_call(f"call-{i:03d}") for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Verify each call has its own state
    call_ids = [r["call_id"] for r in results]
    assert len(set(call_ids)) == 10, "All calls should have unique IDs"

    # Verify no result has turn_count bleeding from another call
    for r in results:
        assert r["turns"] <= 3, f"Call {r['call_id']} has unexpected turn count {r['turns']}"


# ---------------------------------------------------------------------------
# P2: Rapid state cycling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p2_rapid_state_cycling(conductor):
    """Repeatedly cycle: greeting -> qualification -> back via re_engage -> qualification."""
    ctx = conductor.create_context()

    for cycle in range(5):
        # greeting -> qualification
        await conductor.process_turn(ctx, "", pre_classified_intent="confirm_sell")
        assert ctx.current_state == "qualification", f"Cycle {cycle}: expected qualification"

        # qualification -> objection
        await conductor.process_turn(ctx, "", pre_classified_intent="not_interested")
        assert ctx.current_state == "objection_handling", f"Cycle {cycle}: expected objection_handling"

        # objection -> re-engage -> qualification
        await conductor.process_turn(ctx, "", pre_classified_intent="re_engage")
        assert ctx.current_state == "qualification", f"Cycle {cycle}: expected qualification after re_engage"

    assert ctx.phase == ConversationPhase.ACTIVE


# ---------------------------------------------------------------------------
# P3: Max-turns escalation chain
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p3_max_turns_escalation(conductor):
    """Push objection_handling past max_turns, verify escalation or graceful exit."""
    ctx = conductor.create_context()
    ctx.current_state = "objection_handling"
    ctx.state_turn_count = 0

    # objection_handling max_turns=3
    results = []
    for i in range(5):
        r = await conductor.process_turn(ctx, f"still no {i}", pre_classified_intent="general_response")
        results.append(r)
        if ctx.phase != ConversationPhase.ACTIVE:
            break

    # Should have exited or escalated by turn 4
    terminal = any(r.get("blocked") or ctx.phase != ConversationPhase.ACTIVE for r in results)
    assert terminal or ctx.current_state == "live_transfer", \
        f"Expected escalation but got state={ctx.current_state}, phase={ctx.phase}"


# ---------------------------------------------------------------------------
# P4: Full conversation E2E (15 turns)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p4_full_e2e_qualification_flow(conductor):
    """15-turn realistic qualification flow."""
    ctx = conductor.create_context()

    flow = [
        ("confirm_sell", "qualification"),
        ("timeline_question", "qualification"),
        ("condition_question", "qualification"),
        ("motivated_seller", "pricing_discussion"),
        ("price_range_given", "offer_preparation"),
        # offer_preparation fires tools then applies transition via intent
        # tool_complete transitions to offer_presentation
        ("tool_complete", "offer_presentation"),
        ("accept_offer", "closing"),
        ("general_response", "closing"),
        ("no_email", "appointment_setting"),
    ]

    with patch.object(conductor._tool_bridge, "call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"status": "ok", "comps": [], "arv": 350000}

        for i, (intent, expected_state) in enumerate(flow):
            r = await conductor.process_turn(ctx, f"user turn {i}", pre_classified_intent=intent)
            if expected_state is not None:
                assert ctx.current_state == expected_state, \
                    f"Turn {i} intent={intent}: expected {expected_state}, got {ctx.current_state}"

    assert ctx.turn_count == len(flow), f"Expected {len(flow)} turns, got {ctx.turn_count}"


# ---------------------------------------------------------------------------
# P5: Tool timeout simulation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p5_tool_timeout(conductor):
    """Tool timeout should not kill conversation — error is captured."""
    import httpx
    ctx = conductor.create_context()
    ctx.current_state = "offer_preparation"
    ctx.previous_state = "pricing_discussion"

    async def mock_post(*args, **kwargs):
        raise httpx.TimeoutException("timeout")

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        r = await conductor.process_turn(ctx, "ok", pre_classified_intent="general_response")

    # Should have tool_result with error, but not crash
    assert r.get("tool_result") is not None
    assert "error" in r["tool_result"]
    assert ctx.phase == ConversationPhase.ACTIVE  # conversation continues


# ---------------------------------------------------------------------------
# P6: Barge-in during tool call (concurrent)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p6_barge_in_during_tool_call(conductor):
    """Fire two turns concurrently — second should be queued/handled independently."""
    ctx = conductor.create_context()
    ctx.current_state = "offer_preparation"
    ctx.previous_state = "pricing_discussion"

    call_count = 0

    async def slow_tool(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        return {"status": "ok"}

    with patch.object(conductor._tool_bridge, "call", side_effect=slow_tool):
        # Fire both turns concurrently
        results = await asyncio.gather(
            conductor.process_turn(ctx, "go ahead", pre_classified_intent="general_response"),
            conductor.process_turn(ctx, "wait actually", pre_classified_intent="general_response"),
            return_exceptions=True,
        )

    # Both should complete without raising
    for r in results:
        assert not isinstance(r, Exception), f"Barge-in caused exception: {r}"


# ---------------------------------------------------------------------------
# P7: Double handoff — idempotent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p7_double_handoff_idempotent(conductor):
    """Triggering handoff twice should not mutate further or error."""
    ctx = conductor.create_context()
    transfer_calls = []

    async def track_transfer(payload):
        transfer_calls.append(payload)

    ctx.on_transfer = track_transfer

    # First handoff
    r1 = await conductor.process_turn(ctx, "transfer me", pre_classified_intent="transfer")
    assert ctx.phase == ConversationPhase.TRANSFERRED

    # Second attempt — context is exited, should return blocked immediately
    r2 = await conductor.process_turn(ctx, "transfer me again", pre_classified_intent="transfer")
    assert r2["blocked"] is True

    # on_transfer should only fire once (from the first real handoff)
    assert len(transfer_calls) == 1


# ---------------------------------------------------------------------------
# P8: Empty / malformed input
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p8_empty_input(conductor):
    """Empty and whitespace-only inputs should not crash."""
    ctx = conductor.create_context()
    for text in ["", "   ", "\n", "\t"]:
        r = await conductor.process_turn(ctx, text, pre_classified_intent="general_response")
        assert r is not None
        assert ctx.phase == ConversationPhase.ACTIVE


@pytest.mark.asyncio
async def test_p8_unicode_edge_cases(conductor):
    """Unicode and emoji inputs should not crash."""
    ctx = conductor.create_context()
    texts = ["héllo wörld", "こんにちは", "🏠💰🤝", "null\x00byte", "a" * 10000]
    for text in texts:
        r = await conductor.process_turn(ctx, text, pre_classified_intent="general_response")
        assert r is not None


# ---------------------------------------------------------------------------
# P9: Max duration boundary
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p9_max_duration_boundary(conductor):
    """Max duration check fires; last utterance still captured (F4)."""
    ctx = conductor.create_context()
    ctx.start_time = time.time() - 700  # Already past 600s max

    r = await conductor.process_turn(ctx, "Final words before timeout", pre_classified_intent="general_response")

    # F4: Turn must be in transcript
    assert len(ctx.transcript) >= 1
    assert ctx.transcript[0].text == "Final words before timeout"
    assert r["blocked"] is True
    assert ctx.phase == ConversationPhase.TIMED_OUT


# ---------------------------------------------------------------------------
# P10: State machine fuzzer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p10_state_machine_fuzzer(conductor):
    """100 random intents against random states — must not crash."""
    all_states = list(conductor._states.keys())
    all_possible_intents = set()
    for state in conductor._states.values():
        all_possible_intents.update(state.get("transitions", {}).keys())
    all_possible_intents = list(all_possible_intents) or ["general_response"]

    random.seed(42)
    error_count = 0

    with patch.object(conductor._tool_bridge, "call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"status": "ok"}

        for i in range(100):
            ctx = conductor.create_context()
            ctx.current_state = random.choice(all_states)
            intent = random.choice(all_possible_intents)
            try:
                r = await conductor.process_turn(ctx, f"fuzz {i}", pre_classified_intent=intent)
                assert r is not None
            except Exception as e:
                error_count += 1
                print(f"Fuzzer iteration {i}: state={ctx.current_state} intent={intent} error={e}")

    assert error_count == 0, f"Fuzzer found {error_count} crashes"


# ---------------------------------------------------------------------------
# P11: Unknown state recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p11_unknown_state_recovery(conductor):
    """Manually corrupt state — conductor returns blocked, not crash."""
    ctx = conductor.create_context()
    ctx.current_state = "THIS_STATE_DOES_NOT_EXIST_XYZ"

    r = await conductor.process_turn(ctx, "hello", pre_classified_intent="general_response")
    assert r["blocked"] is True
    assert "unknown_state" in r.get("blocked_reason", "")


# ---------------------------------------------------------------------------
# P12: Config with no tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_p12_no_tools_config():
    """Minimal YAML with no tools section works without error."""
    minimal_yaml = """
agent:
  name: "Minimal Agent"
states:
  start:
    phase: "active"
    prompt: "Hello"
    transitions:
      yes: "start"
      no: "start"
"""
    import yaml
    config = yaml.safe_load(minimal_yaml)
    conductor = VoxMaestro(config)
    ctx = conductor.create_context()
    r = await conductor.process_turn(ctx, "yes", pre_classified_intent="yes")
    assert r is not None
    assert ctx.phase == ConversationPhase.ACTIVE
