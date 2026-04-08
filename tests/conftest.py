"""Shared fixtures for VoxMaestro test suite."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from voxmaestro import VoxMaestro
from voxmaestro.conductor import ConversationContext, ConversationPhase


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
RE_AGENT_YAML = EXAMPLES_DIR / "real_estate_agent.yaml"


@pytest.fixture
def re_config() -> dict:
    """Load the real_estate_agent.yaml config."""
    return yaml.safe_load(RE_AGENT_YAML.read_text())


@pytest.fixture
def conductor(re_config) -> VoxMaestro:
    """Create a VoxMaestro instance from the real estate config."""
    return VoxMaestro(re_config)


@pytest.fixture
def ctx(conductor) -> ConversationContext:
    """Create a fresh ConversationContext."""
    return conductor.create_context()


@pytest.fixture
def mock_filler():
    return AsyncMock()


@pytest.fixture
def mock_transfer():
    return AsyncMock()


@pytest.fixture
def mock_metric():
    return AsyncMock()
