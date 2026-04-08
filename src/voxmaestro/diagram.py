"""
VoxMaestro state diagram generator.

Produces Mermaid stateDiagram-v2 markup from a YAML config.

Usage:
    from voxmaestro.diagram import generate_mermaid
    md = generate_mermaid(conductor)
    print(md)
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conductor import VoxMaestro


def generate_mermaid(conductor: "VoxMaestro", highlight_path: list[str] | None = None) -> str:
    """
    Generate Mermaid stateDiagram-v2 from conductor config.

    Args:
        conductor: VoxMaestro instance
        highlight_path: optional list of states to mark as visited (for call replay viz)

    Returns:
        Mermaid diagram string
    """
    lines = ["stateDiagram-v2"]

    highlighted = set(highlight_path or [])
    states = conductor._states

    # State notes (phase labels)
    for name, cfg in states.items():
        has_tools = bool(cfg.get("tools"))
        max_turns = cfg.get("max_turns")

        note_parts = []
        if has_tools:
            note_parts.append("tools")
        if max_turns:
            note_parts.append(f"max:{max_turns}")

        # Reserved for future per-state annotations

    lines.append("")

    # Transitions
    seen_transitions: set[tuple[str, str, str]] = set()
    for state_name, cfg in states.items():
        transitions = cfg.get("transitions", {})
        for raw_intent, target in transitions.items():
            # YAML may parse bare keys like "yes"/"no" as booleans — coerce to str
            intent = str(raw_intent)
            target = str(target)
            if target not in states:
                continue
            key = (state_name, target, intent)
            if key in seen_transitions:
                continue
            seen_transitions.add(key)

            # Truncate long intent labels
            label = intent if len(intent) <= 20 else intent[:17] + "..."
            lines.append(f"    {state_name} --> {target} : {label}")

    lines.append("")

    # Initial state
    first_state = list(states.keys())[0]
    lines.append(f"    [*] --> {first_state}")

    # Terminal states (no outgoing transitions or only empty)
    for state_name, cfg in states.items():
        transitions = cfg.get("transitions", {})
        if not transitions:
            lines.append(f"    {state_name} --> [*]")

    # Highlight visited states
    if highlighted:
        lines.append("")
        lines.append("    classDef visited fill:#ff6b6b,color:#fff")
        visited_list = ",".join(highlighted)
        lines.append(f"    class {visited_list} visited")

    return "\n".join(lines)


def generate_mermaid_html(conductor: "VoxMaestro", highlight_path: list[str] | None = None) -> str:
    """Wrap Mermaid diagram in a minimal HTML page for browser viewing."""
    diagram = generate_mermaid(conductor, highlight_path)
    return f"""<!DOCTYPE html>
<html>
<head>
<title>{conductor._agent.get('name', 'VoxMaestro')} — State Diagram</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
</head>
<body style="background:#1a1a2e;color:#eee;font-family:sans-serif;padding:2em">
<h2>{conductor._agent.get('name', 'VoxMaestro')} State Machine</h2>
<div class="mermaid">
{diagram}
</div>
<script>mermaid.initialize({{startOnLoad:true,theme:'dark'}});</script>
</body>
</html>"""
