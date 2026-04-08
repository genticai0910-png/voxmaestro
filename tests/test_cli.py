"""Tests for VoxMaestro CLI (__main__.py)."""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


class TestScoreCommand:
    def test_score_from_file(self, tmp_path):
        """Score command reads JSON file and outputs analysis."""
        payload = {
            "call_id": "bland-cli-001",
            "duration": 90.0,
            "transcript": [
                {"role": "assistant", "text": "Hi there"},
                {"role": "user", "text": "yes I want to sell"},
                {"role": "user", "text": "I'm very motivated"},
            ]
        }
        f = tmp_path / "call.json"
        f.write_text(json.dumps(payload))

        import asyncio
        from voxmaestro.__main__ import cmd_score

        class FakeArgs:
            file = str(f)
            yaml = str(EXAMPLES_DIR / "real_estate_agent.yaml")

        result = asyncio.run(cmd_score(FakeArgs()))
        assert "voice_score" in result
        assert "voice_tier" in result
        assert result["bland_call_id"] == "bland-cli-001"

    def test_score_empty_transcript(self, tmp_path):
        payload = {"call_id": "empty-001", "transcript": []}
        f = tmp_path / "empty.json"
        f.write_text(json.dumps(payload))

        import asyncio
        from voxmaestro.__main__ import cmd_score

        class FakeArgs:
            file = str(f)
            yaml = str(EXAMPLES_DIR / "real_estate_agent.yaml")

        result = asyncio.run(cmd_score(FakeArgs()))
        assert result["voice_score"] == 0
        assert result["turns_processed"] == 0

    def test_find_yaml_explicit_path(self):
        from voxmaestro.__main__ import find_yaml
        p = find_yaml(str(EXAMPLES_DIR / "real_estate_agent.yaml"))
        assert p.exists()

    def test_find_yaml_not_found(self):
        from voxmaestro.__main__ import find_yaml
        with pytest.raises(FileNotFoundError):
            find_yaml("/definitely/does/not/exist.yaml")

    def test_find_yaml_auto_discover(self):
        """Auto-discovery finds the bundled example YAML."""
        from voxmaestro.__main__ import find_yaml
        p = find_yaml(None)
        assert p.exists()
