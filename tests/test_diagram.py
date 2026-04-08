"""Tests for state diagram generation."""
import pytest
from pathlib import Path
from voxmaestro import VoxMaestro
from voxmaestro.diagram import generate_mermaid, generate_mermaid_html

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def conductor():
    return VoxMaestro.from_yaml(EXAMPLES_DIR / "real_estate_agent.yaml")


class TestMermaidDiagram:
    def test_generates_mermaid(self, conductor):
        md = generate_mermaid(conductor)
        assert "stateDiagram-v2" in md
        assert "greeting" in md
        assert "-->" in md

    def test_all_states_present(self, conductor):
        md = generate_mermaid(conductor)
        for state in conductor._states:
            assert state in md, f"State '{state}' missing from diagram"

    def test_initial_state_arrow(self, conductor):
        md = generate_mermaid(conductor)
        first = list(conductor._states.keys())[0]
        assert f"[*] --> {first}" in md

    def test_highlight_path(self, conductor):
        path = ["greeting", "qualification"]
        md = generate_mermaid(conductor, highlight_path=path)
        assert "classDef visited" in md
        assert "greeting" in md

    def test_html_output(self, conductor):
        html = generate_mermaid_html(conductor)
        assert "<html>" in html
        assert "mermaid" in html
        assert "stateDiagram" in html

    def test_minimal_config(self):
        import yaml
        cfg = yaml.safe_load("""
agent:
  name: Mini
states:
  start:
    transitions:
      yes: end
  end:
    transitions: {}
""")
        c = VoxMaestro(cfg)
        md = generate_mermaid(c)
        assert "start --> end" in md

    def test_no_invalid_state_transitions(self, conductor):
        """Transitions to non-existent states should be excluded."""
        md = generate_mermaid(conductor)
        lines = md.split("\n")
        # Extract all transition targets
        for line in lines:
            if "-->" in line and "[*]" not in line:
                parts = line.strip().split("-->")
                if len(parts) == 2:
                    target = parts[1].split(":")[0].strip()
                    assert target in conductor._states, \
                        f"Diagram references unknown state: {target}"

    def test_highlight_classDef_format(self, conductor):
        path = list(conductor._states.keys())[:2]
        md = generate_mermaid(conductor, highlight_path=path)
        assert "classDef visited fill:#ff6b6b,color:#fff" in md
        # All highlighted states listed in class directive
        class_line = [l for l in md.split("\n") if l.strip().startswith("class ") and "visited" in l]
        assert len(class_line) == 1
        for state in path:
            assert state in class_line[0]

    def test_long_intent_truncated(self):
        import yaml
        cfg = yaml.safe_load("""
agent:
  name: Test
states:
  start:
    transitions:
      this_is_a_very_long_intent_name_that_exceeds_twenty_chars: end
  end:
    transitions: {}
""")
        c = VoxMaestro(cfg)
        md = generate_mermaid(c)
        # Label must be truncated: intent[:17] + "..." = 20 chars total
        assert "this_is_a_very_lo..." in md

    def test_terminal_state_arrow(self):
        import yaml
        cfg = yaml.safe_load("""
agent:
  name: Test
states:
  start:
    transitions:
      go: terminal
  terminal:
    transitions: {}
""")
        c = VoxMaestro(cfg)
        md = generate_mermaid(c)
        assert "terminal --> [*]" in md

    def test_html_contains_agent_name(self, conductor):
        html = generate_mermaid_html(conductor)
        agent_name = conductor._agent.get("name", "VoxMaestro")
        assert agent_name in html
