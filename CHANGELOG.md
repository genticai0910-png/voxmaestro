# Changelog

All notable changes to VoxMaestro are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] — 2026-04-08

### Added
- **FastAPI microservice** (`voxmaestro.server`) — standalone HTTP server on port 8850
  - `GET /health`, `GET /diagram`, `GET /diagram.html`
  - `POST /replay`, `POST /score`, `POST /live-turn`, `POST /analyze`
  - `GET /analytics` — in-memory funnel report
  - Bearer token auth via `VOX_API_KEY` env var (optional)
- **`serve` CLI command** — `python -m voxmaestro serve --port 8850`
- **LaunchAgent plist** for Mac Mini deployment (`com.voxmaestro.server.plist`)
- **Analytics module** (`voxmaestro.analytics.CallFunnelAnalyzer`)
  - Score distribution, tier breakdown, state reach rates
  - Avg turns to each state, top exit intents, conversion/handoff rates
- **State diagram generator** (`voxmaestro.diagram`)
  - `generate_mermaid()` — Mermaid stateDiagram-v2 output
  - `generate_mermaid_html()` — browser-viewable HTML page
  - `diagram` CLI command with `--html` and `--output` flags
- **Training data harvester** (`voxmaestro.training.TrainingHarvester`)
  - Per-turn labeled example collection from replays and live turns
  - JSONL persistence with daily rotation
  - Alpaca format export for fine-tuning pipelines
- **iRELOP integration** (`voxmaestro.integrations.irelop`)
  - `VoxIRELOPEnricher` — maps `CallAnalysis` to motivation bonus + timeline signals
  - `VoiceSignals` dataclass — structured iRELOP-compatible output
  - `enrich_and_post()` — async POST to n8n lead-enrich webhook
- **Bland live-turn handler** (`BlandLiveTurnHandler`)
  - Per-call session state with 30-min TTL auto-expiry
  - State-specific Bland agent directives
  - `end_session()` for explicit cleanup
- **`dealiq-ce` intent provider** — Conversational Extraction model (MLX `:11436`)
  - JSON-structured intent extraction with confidence score
  - Falls back cleanly on parse errors
- **`stats` CLI command** — aggregate funnel report across a directory of call JSON files
- **`transcript_replay()` method** on `VoxMaestro` — batch replay list of utterances

### Fixed
- **F1**: `tool_call` state auto-returns via `return_to: previous` — conductor was ignoring this field
- **F2**: Per-call callbacks on `ConversationContext` — `on_filler`/`on_transfer`/`on_metric` no longer shared across concurrent calls
- **F4**: User turn recorded before max-duration check — last utterance no longer lost on timeout
- **F5**: `apply_transition` skipped after handoff/exit — no state mutation after `TRANSFERRED`
- **F6**: HTTP client injection — `set_http_client()` + `close()` on `ToolBridge` and `IntentClassifier`
- **F10**: JSON Schema validation on YAML load via `jsonschema` (optional dep)
- **F13**: Tool call retry with exponential backoff (configurable `retry` field per tool)
- Self-loop transitions no longer reset `state_turn_count` — max_turns now accumulates correctly
- YAML boolean coercion in state diagram generator (`yes`/`no` keys parsed as booleans by PyYAML)

### Changed
- `real_estate_agent.yaml` intent provider updated to `dealiq-ce` (MLX CE model at `:11436`)
- `BlandTranscriptAdapter.replay()` stores turn texts in `analysis.metadata` for training harvest

## [0.1.0] — 2026-04-08

### Added
- Initial release
- YAML-defined conversation state machine (`VoxMaestro`, `ConversationContext`)
- `SchemaLoader` with JSON Schema validation
- `ToolBridge` — async HTTP tool calls with retry/backoff
- `IntentClassifier` — Ollama/OpenAI-compatible intent classification
- `BlandTranscriptAdapter` — post-call transcript replay and `CallAnalysis`
- `qualification_score()` — 0-100 voice depth scoring
- CLI: `replay`, `score` commands
- 13 bug fixes from pressure testing
- 55 tests across core conductor, Bland adapter, and pressure suite
- GitHub Actions CI (Python 3.10/3.11/3.12 matrix)
- Apache-2.0 license
