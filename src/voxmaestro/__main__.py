"""
VoxMaestro CLI.

Usage:
    python -m voxmaestro replay --call-id <id> [--bland-api-key KEY] [--yaml PATH]
    python -m voxmaestro score --file <path>  [--yaml PATH]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


DEFAULT_YAML = Path(__file__).parent.parent.parent / "examples" / "real_estate_agent.yaml"
# Also try relative to CWD
FALLBACK_YAML = Path("examples/real_estate_agent.yaml")


def find_yaml(provided: str | None) -> Path:
    if provided:
        p = Path(provided)
        if p.exists():
            return p
        raise FileNotFoundError(f"YAML not found: {provided}")
    for candidate in [DEFAULT_YAML, FALLBACK_YAML, Path.home() / "voxmaestro/examples/real_estate_agent.yaml"]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate real_estate_agent.yaml — pass --yaml PATH")


async def cmd_replay(args):
    import httpx
    from voxmaestro import VoxMaestro
    from voxmaestro.integrations.bland import BlandTranscriptAdapter
    from voxmaestro.integrations.irelop import VoxIRELOPEnricher

    api_key = args.bland_api_key or os.environ.get("BLAND_API_KEY", "")
    if not api_key:
        print("ERROR: --bland-api-key or BLAND_API_KEY env var required", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching call {args.call_id} from Bland API...")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"https://api.bland.ai/v1/calls/{args.call_id}",
            headers={"authorization": api_key},
        )
        resp.raise_for_status()
        payload = resp.json()

    yaml_path = find_yaml(args.yaml)
    conductor = VoxMaestro.from_yaml(yaml_path)
    adapter = BlandTranscriptAdapter(conductor)
    enricher = VoxIRELOPEnricher()

    print(f"Replaying transcript through {yaml_path.name}...")
    analysis = await adapter.replay(payload)
    signals = enricher.extract_signals(analysis)

    from voxmaestro.integrations.bland import qualification_score
    score = qualification_score(analysis)

    print("\n" + "=" * 60)
    print(f"  VoxMaestro Call Analysis — {args.call_id}")
    print("=" * 60)
    print(f"  Final State:     {analysis.final_state}")
    print(f"  Phase:           {analysis.phase}")
    print(f"  Turns Processed: {analysis.turns_processed}/{analysis.transcript_turns}")
    print(f"  Duration:        {int(analysis.duration_seconds or 0)}s")
    print(f"  Voice Score:     {score}/100  [{signals.voice_tier}]")
    print(f"  State Path:      {' → '.join(analysis.state_path)}")
    print(f"  Intents:         {', '.join(analysis.intents) or 'none'}")
    print(f"\n  iRELOP Signals:")
    print(f"    Motivation bonus:  +{signals.motivation_bonus}pts")
    if signals.timeline_days:
        print(f"    Timeline:          {signals.timeline_days}d")
    else:
        print(f"    Timeline:          unknown")
    print(f"    Distress:          {', '.join(signals.distress_signals) or 'none'}")
    print(f"    Decision maker:    {signals.is_decision_maker}")
    print()

    if analysis.errors:
        print(f"  Errors: {analysis.errors}")

    return analysis


async def cmd_score(args):
    from voxmaestro import VoxMaestro
    from voxmaestro.integrations.bland import BlandTranscriptAdapter
    from voxmaestro.integrations.irelop import VoxIRELOPEnricher
    from voxmaestro.integrations.bland import qualification_score

    payload = json.loads(Path(args.file).read_text())
    yaml_path = find_yaml(args.yaml)
    conductor = VoxMaestro.from_yaml(yaml_path)
    adapter = BlandTranscriptAdapter(conductor)
    enricher = VoxIRELOPEnricher()

    analysis = await adapter.replay(payload)
    signals = enricher.extract_signals(analysis)
    score = qualification_score(analysis)

    output = {
        "call_id": analysis.call_id,
        "bland_call_id": analysis.bland_call_id,
        "final_state": analysis.final_state,
        "phase": analysis.phase,
        "voice_score": score,
        "voice_tier": signals.voice_tier,
        "state_path": analysis.state_path,
        "intents": analysis.intents,
        "turns_processed": analysis.turns_processed,
        "qualification_reached": analysis.qualification_reached,
        "pricing_reached": analysis.pricing_reached,
        "offer_reached": analysis.offer_reached,
        "handoff_triggered": analysis.handoff_triggered,
        "irelop_signals": {
            "motivation_bonus": signals.motivation_bonus,
            "timeline_days": signals.timeline_days,
            "distress_signals": signals.distress_signals,
            "condition_hint": signals.condition_hint,
            "is_decision_maker": signals.is_decision_maker,
        },
    }

    print(json.dumps(output, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(prog="voxmaestro", description="VoxMaestro CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_replay = sub.add_parser("replay", help="Fetch a Bland call and replay through state machine")
    p_replay.add_argument("--call-id", required=True, help="Bland call ID")
    p_replay.add_argument("--bland-api-key", default="", help="Bland API key (or set BLAND_API_KEY)")
    p_replay.add_argument("--yaml", default="", help="Path to agent YAML config")

    p_score = sub.add_parser("score", help="Score a Bland post-call JSON payload file")
    p_score.add_argument("--file", required=True, help="Path to Bland post-call JSON file")
    p_score.add_argument("--yaml", default="", help="Path to agent YAML config")

    args = parser.parse_args()

    if args.command == "replay":
        asyncio.run(cmd_replay(args))
    elif args.command == "score":
        asyncio.run(cmd_score(args))


if __name__ == "__main__":
    main()
