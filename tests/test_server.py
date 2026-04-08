"""Tests for VoxMaestro HTTP server."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


@pytest.fixture
def test_app():
    """Create FastAPI test client."""
    import os
    os.environ["VOX_AGENT_YAML"] = str(EXAMPLES_DIR / "real_estate_agent.yaml")
    from fastapi.testclient import TestClient
    from voxmaestro.server import app
    with TestClient(app) as client:
        yield client


class TestServerHealth:
    def test_health(self, test_app):
        r = test_app.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["conductor"] == "DealiQ RE Agent"
        assert data["states"] > 0

    def test_diagram_endpoint(self, test_app):
        r = test_app.get("/diagram")
        assert r.status_code == 200
        assert "stateDiagram-v2" in r.text
        assert "greeting" in r.text

    def test_diagram_html(self, test_app):
        r = test_app.get("/diagram.html")
        assert r.status_code == 200
        assert "<html>" in r.text


class TestReplayEndpoint:
    def test_replay_basic(self, test_app):
        r = test_app.post("/replay", json={
            "call_id": "server-test-001",
            "transcript": [
                {"role": "user", "text": "yes sell"},
            ],
        })
        assert r.status_code == 200
        data = r.json()
        assert "score" in data
        assert "tier" in data
        assert "state_path" in data
        assert "latency_seconds" in data

    def test_replay_empty(self, test_app):
        r = test_app.post("/replay", json={"transcript": []})
        assert r.status_code == 200
        data = r.json()
        assert data["turns_processed"] == 0

    def test_score_alias(self, test_app):
        r = test_app.post("/score", json={
            "transcript": [{"role": "user", "text": "not interested"}],
        })
        assert r.status_code == 200
        assert "score" in r.json()


class TestLiveTurnEndpoint:
    def test_live_turn(self, test_app):
        r = test_app.post("/live-turn", json={
            "call_id": "live-server-001",
            "transcript": [
                {"role": "assistant", "text": "Hi"},
                {"role": "user", "text": "yes I want to sell"},
            ]
        })
        assert r.status_code == 200
        assert "response" in r.json()

    def test_live_turn_empty(self, test_app):
        r = test_app.post("/live-turn", json={"call_id": "x", "transcript": []})
        assert r.status_code == 200
        assert r.json() == {"response": ""}


class TestAnalyzeEndpoint:
    def test_analyze_basic(self, test_app):
        r = test_app.post("/analyze", json={
            "call_id": "analyze-001",
            "transcript": [{"role": "user", "text": "yes sell"}],
        })
        assert r.status_code == 200
        data = r.json()
        assert "score" in data
        assert "signals" in data
        assert "lead_data_patch" in data
        assert "irelop_fired" in data

    def test_analytics_accumulates(self, test_app):
        # Make two calls
        for i in range(3):
            test_app.post("/replay", json={"transcript": [{"role": "user", "text": "sell"}]})
        r = test_app.get("/analytics")
        assert r.status_code == 200
        data = r.json()
        assert data["total_calls"] >= 3


class TestAuth:
    def test_no_auth_key_set_allows_all(self, test_app):
        """When VOX_API_KEY not set, all requests pass."""
        r = test_app.get("/diagram")
        assert r.status_code == 200

    @pytest.mark.skip(reason="Module reload across test isolation is unreliable in same process — test manually via curl")
    def test_auth_key_enforced(self, tmp_path):
        import os
        os.environ["VOX_API_KEY"] = "test-key-xyz"
        os.environ["VOX_AGENT_YAML"] = str(EXAMPLES_DIR / "real_estate_agent.yaml")
        from fastapi.testclient import TestClient
        # Re-import to pick up new env
        import importlib
        import voxmaestro.server as srv
        importlib.reload(srv)
        with TestClient(srv.app) as client:
            # Without key → 401 or 403
            r = client.get("/diagram")
            assert r.status_code in (401, 403)
            # With key → 200
            r2 = client.get("/diagram", headers={"Authorization": "Bearer test-key-xyz"})
            assert r2.status_code == 200
        del os.environ["VOX_API_KEY"]
