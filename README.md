# VoxMaestro

YAML-defined voice agent conductor — state machine with mid-turn tool bridging, 3-phase handoff protocol, and pluggable intent classification.

Extracted from the DealiQ/Gentic voice stack.

## Features

- **YAML state machines** — define conversation flows declaratively
- **Intent classification** — provider-agnostic (Ollama, MLX, OpenAI-compatible)
- **Tool bridge** — async HTTP tool calls with retry/backoff
- **3-phase handoff** — graceful exit, live transfer, timeout handling
- **Per-call contexts** — safe concurrent use of shared conductor
- **Langfuse observability** — optional trace + span instrumentation
- **Pipecat integration** — frame processor adapter (when Pipecat is deployed)

## Install

```bash
pip install voxmaestro
# With Langfuse observability:
pip install "voxmaestro[langfuse]"
```

## Quick Start

```python
from voxmaestro import VoxMaestro

conductor = VoxMaestro.from_yaml("examples/real_estate_agent.yaml")
ctx = conductor.create_context()

result = await conductor.process_turn(
    ctx,
    "Yes, I'm interested in selling my house",
    pre_classified_intent="confirm_sell",  # or let VoxMaestro classify
)
print(result["state"])   # → "qualification"
print(result["intent"])  # → "confirm_sell"
```

## Configuration

See `examples/real_estate_agent.yaml` for a full DealiQ real estate agent config.

Key sections:
- `agent` — name, max_duration_seconds
- `intent` — Ollama endpoint + model for classification
- `states` — state definitions with transitions, tools, filler
- `tools` — HTTP tool endpoints with retry config
- `handoff` — webhook channels for call completion
- `observability` — Langfuse config

## Architecture

```
VoxMaestro (shared, stateless)
  ├── SchemaLoader       — YAML load + JSON Schema validation
  ├── IntentClassifier   — Ollama/OpenAI intent classification
  ├── ToolBridge         — async HTTP tool calls with retry
  └── per-call ConversationContext (isolated state per call)
```

## License

Apache-2.0
