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


async def cmd_stats(args):
    from voxmaestro import VoxMaestro, CallFunnelAnalyzer
    from voxmaestro.integrations.bland import BlandTranscriptAdapter

    yaml_path = find_yaml(args.yaml)
    conductor = VoxMaestro.from_yaml(yaml_path)
    adapter = BlandTranscriptAdapter(conductor)
    analyzer = CallFunnelAnalyzer()

    scan_dir = Path(args.dir)
    if not scan_dir.is_dir():
        print(f"ERROR: --dir must be a directory, got: {args.dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(scan_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files found in {scan_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(json_files)} call file(s) from {scan_dir}...")
    errors = 0
    for fpath in json_files:
        try:
            payload = json.loads(fpath.read_text())
            analysis = await adapter.replay(payload)
            analyzer.ingest_one(analysis)
        except Exception as e:
            print(f"  WARN: {fpath.name} skipped — {e}", file=sys.stderr)
            errors += 1

    report = analyzer.report()
    print("\n" + "=" * 60)
    print(f"  VoxMaestro Funnel Report — {report.total_calls} call(s)")
    print("=" * 60)
    print(f"  Avg Score:          {report.avg_score}/100")
    print(f"  Conversion Rate:    {report.conversion_rate * 100:.1f}%")
    print(f"  Handoff Rate:       {report.handoff_rate * 100:.1f}%")
    print(f"  Avg Turns:          {report.avg_turns}")
    if report.avg_duration_seconds is not None:
        print(f"  Avg Duration:       {report.avg_duration_seconds}s")
    print(f"\n  Score Distribution:")
    for bucket, count in report.score_distribution.items():
        print(f"    {bucket:>8}: {count}")
    print(f"\n  Tier Distribution:")
    for tier, count in report.tier_distribution.items():
        print(f"    {tier:>6}: {count}")
    print(f"\n  State Reach Rates:")
    for state, rate in sorted(report.state_reach_rates.items(), key=lambda x: -x[1]):
        print(f"    {state:<30} {rate * 100:.1f}%")
    if report.top_exit_intents:
        print(f"\n  Top Exit Intents:")
        for intent, count in report.top_exit_intents[:5]:
            print(f"    {intent:<30} {count}")
    if errors:
        print(f"\n  Skipped: {errors} file(s) with errors", file=sys.stderr)
    print()


async def cmd_diagram(args):
    from voxmaestro import VoxMaestro
    from voxmaestro.diagram import generate_mermaid, generate_mermaid_html

    yaml_path = find_yaml(args.yaml)
    conductor = VoxMaestro.from_yaml(yaml_path)

    if args.html:
        content = generate_mermaid_html(conductor)
    else:
        content = generate_mermaid(conductor)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(content)
        print(f"Written to {out_path}")
    else:
        print(content)


def cmd_serve(args):
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install 'voxmaestro[server]'", file=sys.stderr)
        sys.exit(1)
    if args.yaml:
        os.environ.setdefault("VOX_AGENT_YAML", args.yaml)
    if args.port:
        os.environ.setdefault("VOX_PORT", str(args.port))
    uvicorn.run(
        "voxmaestro.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


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

    p_stats = sub.add_parser("stats", help="Aggregate funnel report from a directory of JSON call files")
    p_stats.add_argument("--dir", required=True, help="Directory containing .json Bland call files")
    p_stats.add_argument("--yaml", default="", help="Path to agent YAML config")

    p_diagram = sub.add_parser("diagram", help="Generate Mermaid state diagram from YAML config")
    p_diagram.add_argument("--yaml", default="", help="Path to agent YAML config")
    p_diagram.add_argument("--output", default="", help="Write output to this file instead of stdout")
    p_diagram.add_argument("--html", action="store_true", help="Output full HTML page instead of Mermaid markup")

    p_serve = sub.add_parser("serve", help="Start VoxMaestro HTTP server")
    p_serve.add_argument("--port", type=int, default=8850)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--yaml", default="")
    p_serve.add_argument("--reload", action="store_true", help="Hot reload (dev only)")

    args = parser.parse_args()

    if args.command == "replay":
        asyncio.run(cmd_replay(args))
    elif args.command == "score":
        asyncio.run(cmd_score(args))
    elif args.command == "stats":
        asyncio.run(cmd_stats(args))
    elif args.command == "diagram":
        asyncio.run(cmd_diagram(args))
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
