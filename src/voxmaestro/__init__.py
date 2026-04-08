"""VoxMaestro — voice agent conductor engine."""
from .conductor import (
    VoxMaestro,
    ConversationContext,
    ConversationPhase,
    ToolBridge,
)
from .integrations.bland import BlandTranscriptAdapter, BlandLiveTurnHandler, CallAnalysis, qualification_score
from .integrations.irelop import VoxIRELOPEnricher, VoiceSignals
from .training import TrainingHarvester, TrainingExample
from .analytics import CallFunnelAnalyzer

__all__ = [
    "VoxMaestro",
    "ConversationContext",
    "ConversationPhase",
    "ToolBridge",
    "BlandTranscriptAdapter",
    "BlandLiveTurnHandler",
    "CallAnalysis",
    "qualification_score",
    "VoxIRELOPEnricher",
    "VoiceSignals",
    "TrainingHarvester",
    "TrainingExample",
    "CallFunnelAnalyzer",
]
__version__ = "0.2.0"
