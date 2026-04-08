"""
VoxMaestro analytics — aggregate call scoring and funnel analysis.

Usage:
    from voxmaestro.analytics import CallFunnelAnalyzer
    analyzer = CallFunnelAnalyzer()
    analyzer.ingest(analysis_list)
    report = analyzer.report()
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .integrations.bland import CallAnalysis


@dataclass
class FunnelReport:
    total_calls: int
    avg_score: float
    score_distribution: dict[str, int]    # "0-24", "25-49", "50-74", "75-100" → count
    tier_distribution: dict[str, int]     # HOT/WARM/COOL/PASS → count
    state_reach_rates: dict[str, float]   # state → % of calls that reached it
    avg_turns_to_state: dict[str, float]  # state → avg turns to first reach
    top_exit_intents: list[tuple[str, int]]  # [(intent, count)] sorted desc
    avg_turns: float
    avg_duration_seconds: float | None
    conversion_rate: float                # % that reached pricing_discussion or deeper
    handoff_rate: float
    errors: list[str]


class CallFunnelAnalyzer:
    def __init__(self):
        self._analyses: list["CallAnalysis"] = []

    def ingest(self, analyses: list["CallAnalysis"]) -> None:
        self._analyses.extend(analyses)

    def ingest_one(self, analysis: "CallAnalysis") -> None:
        self._analyses.append(analysis)

    def report(self) -> FunnelReport:
        from .integrations.bland import qualification_score

        if not self._analyses:
            return FunnelReport(
                total_calls=0, avg_score=0.0, score_distribution={},
                tier_distribution={}, state_reach_rates={}, avg_turns_to_state={},
                top_exit_intents=[], avg_turns=0.0, avg_duration_seconds=None,
                conversion_rate=0.0, handoff_rate=0.0, errors=[],
            )

        scores = [qualification_score(a) for a in self._analyses]
        avg_score = sum(scores) / len(scores)

        # Score distribution buckets
        score_dist: dict[str, int] = {"0-24": 0, "25-49": 0, "50-74": 0, "75-100": 0}
        tier_dist: dict[str, int] = {"HOT": 0, "WARM": 0, "COOL": 0, "PASS": 0}
        for s in scores:
            if s >= 75:   score_dist["75-100"] += 1
            elif s >= 50: score_dist["50-74"] += 1
            elif s >= 25: score_dist["25-49"] += 1
            else:         score_dist["0-24"] += 1
            if s >= 80:   tier_dist["HOT"] += 1
            elif s >= 60: tier_dist["WARM"] += 1
            elif s >= 40: tier_dist["COOL"] += 1
            else:         tier_dist["PASS"] += 1

        # State reach rates
        all_states: set[str] = set()
        for a in self._analyses:
            all_states.update(a.state_path)
        state_reach: dict[str, int] = {s: 0 for s in all_states}
        turns_to_state: dict[str, list[int]] = defaultdict(list)
        for a in self._analyses:
            seen: set[str] = set()
            for i, state in enumerate(a.state_path):
                if state not in seen:
                    state_reach[state] += 1
                    turns_to_state[state].append(i)
                    seen.add(state)

        n = len(self._analyses)
        state_reach_rates = {s: round(c / n, 3) for s, c in state_reach.items()}
        avg_turns_to_state = {
            s: round(sum(turns) / len(turns), 1)
            for s, turns in turns_to_state.items() if turns
        }

        # Exit intents — last intent of each call
        exit_intents = [a.intents[-1] for a in self._analyses if a.intents]
        top_exits = Counter(exit_intents).most_common(10)

        # Averages
        avg_turns = sum(a.turns_processed for a in self._analyses) / n
        durations = [a.duration_seconds for a in self._analyses if a.duration_seconds]
        avg_dur = sum(durations) / len(durations) if durations else None

        # Rates
        converted = sum(1 for a in self._analyses if a.pricing_reached or a.offer_reached)
        handoffs = sum(1 for a in self._analyses if a.handoff_triggered)

        return FunnelReport(
            total_calls=n,
            avg_score=round(avg_score, 1),
            score_distribution=score_dist,
            tier_distribution=tier_dist,
            state_reach_rates=state_reach_rates,
            avg_turns_to_state=avg_turns_to_state,
            top_exit_intents=top_exits,
            avg_turns=round(avg_turns, 1),
            avg_duration_seconds=round(avg_dur, 1) if avg_dur else None,
            conversion_rate=round(converted / n, 3),
            handoff_rate=round(handoffs / n, 3),
            errors=[],
        )

    def to_json(self) -> str:
        r = self.report()
        return json.dumps({
            "total_calls": r.total_calls,
            "avg_score": r.avg_score,
            "score_distribution": r.score_distribution,
            "tier_distribution": r.tier_distribution,
            "state_reach_rates": r.state_reach_rates,
            "avg_turns_to_state": r.avg_turns_to_state,
            "top_exit_intents": [{"intent": i, "count": c} for i, c in r.top_exit_intents],
            "avg_turns": r.avg_turns,
            "avg_duration_seconds": r.avg_duration_seconds,
            "conversion_rate": r.conversion_rate,
            "handoff_rate": r.handoff_rate,
        }, indent=2)
