"""Tests for call funnel analytics."""
import pytest
from pathlib import Path
from voxmaestro import VoxMaestro, CallFunnelAnalyzer
from voxmaestro.integrations.bland import BlandTranscriptAdapter, CallAnalysis

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def make_analysis(score_tier, **kwargs):
    """Helper to create a CallAnalysis at a given depth."""
    from voxmaestro.integrations.bland import CallAnalysis
    tiers = {
        "greeting": dict(state_path=["greeting"], intents=["not_interested"],
                        qualification_reached=False, pricing_reached=False, offer_reached=False,
                        handoff_triggered=False, turns_processed=1, final_state="greeting"),
        "qualification": dict(state_path=["greeting","qualification"], intents=["confirm_sell"],
                             qualification_reached=True, pricing_reached=False, offer_reached=False,
                             handoff_triggered=False, turns_processed=2, final_state="qualification"),
        "pricing": dict(state_path=["greeting","qualification","pricing_discussion"],
                       intents=["confirm_sell","motivated_seller"],
                       qualification_reached=True, pricing_reached=True, offer_reached=False,
                       handoff_triggered=False, turns_processed=3, final_state="pricing_discussion"),
        "offer": dict(state_path=["greeting","qualification","pricing_discussion","offer_presentation"],
                     intents=["confirm_sell","motivated_seller","price_range_given"],
                     qualification_reached=True, pricing_reached=True, offer_reached=True,
                     handoff_triggered=False, turns_processed=4, final_state="offer_presentation"),
    }
    base = dict(call_id="x", bland_call_id="b", phase="active",
                duration_seconds=60.0, metadata={}, errors=[],
                handoff_reason=None, transcript_turns=0)
    base.update(tiers[score_tier])
    base.update(kwargs)
    return CallAnalysis(**base)


class TestCallFunnelAnalyzer:
    def test_empty_report(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        r = a.report()
        assert r.total_calls == 0
        assert r.avg_score == 0.0

    def test_single_call(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        a.ingest_one(make_analysis("qualification"))
        r = a.report()
        assert r.total_calls == 1
        assert r.avg_score > 0

    def test_tier_distribution(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        a.ingest([
            make_analysis("greeting"),       # PASS
            make_analysis("qualification"),  # COOL or WARM
            make_analysis("pricing"),        # WARM
            make_analysis("offer"),          # HOT or WARM
        ])
        r = a.report()
        assert r.total_calls == 4
        total_tiered = sum(r.tier_distribution.values())
        assert total_tiered == 4

    def test_conversion_rate(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        a.ingest([
            make_analysis("greeting"),
            make_analysis("pricing"),
            make_analysis("offer"),
        ])
        r = a.report()
        assert r.conversion_rate == pytest.approx(2/3, rel=0.01)

    def test_state_reach_rates(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        a.ingest([
            make_analysis("greeting"),
            make_analysis("qualification"),
            make_analysis("qualification"),
        ])
        r = a.report()
        assert "greeting" in r.state_reach_rates
        assert r.state_reach_rates["greeting"] == 1.0  # all 3 calls reach greeting
        assert r.state_reach_rates.get("qualification", 0) == pytest.approx(2/3, rel=0.01)

    def test_to_json(self):
        import json
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        a.ingest_one(make_analysis("pricing"))
        j = json.loads(a.to_json())
        assert j["total_calls"] == 1
        assert "tier_distribution" in j

    def test_score_distribution_buckets(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        # greeting=0, qualification=25, pricing=50, offer=75
        a.ingest([
            make_analysis("greeting"),
            make_analysis("qualification"),
            make_analysis("pricing"),
            make_analysis("offer"),
        ])
        r = a.report()
        assert r.score_distribution["0-24"] == 1
        assert r.score_distribution["25-49"] == 1
        assert r.score_distribution["50-74"] == 1
        assert r.score_distribution["75-100"] == 1

    def test_handoff_rate(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        a.ingest([
            make_analysis("greeting"),
            make_analysis("greeting", handoff_triggered=True),
            make_analysis("pricing"),
        ])
        r = a.report()
        assert r.handoff_rate == pytest.approx(1/3, rel=0.01)

    def test_avg_turns(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        a.ingest([
            make_analysis("greeting"),    # turns_processed=1
            make_analysis("qualification"),  # turns_processed=2
            make_analysis("pricing"),     # turns_processed=3
        ])
        r = a.report()
        assert r.avg_turns == pytest.approx(2.0, rel=0.01)

    def test_top_exit_intents(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        a.ingest([
            make_analysis("greeting"),    # last intent: not_interested
            make_analysis("greeting"),    # last intent: not_interested
            make_analysis("qualification"),  # last intent: confirm_sell
        ])
        r = a.report()
        top_intent, top_count = r.top_exit_intents[0]
        assert top_intent == "not_interested"
        assert top_count == 2

    def test_ingest_accumulates(self):
        from voxmaestro.analytics import CallFunnelAnalyzer
        a = CallFunnelAnalyzer()
        a.ingest_one(make_analysis("greeting"))
        a.ingest([make_analysis("pricing"), make_analysis("offer")])
        r = a.report()
        assert r.total_calls == 3
