# Changelog

All notable changes to VoxMaestro are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] — 2026-04-09

### Added
- Initial open source release
- YAML-defined conversation state machine (`VoxMaestro`, `ConversationContext`)
- `SchemaLoader` with JSON Schema validation (`schema.json`)
- `StateMachine` — guarded transitions, max_turns escalation, wildcard routing
- `ToolBridge` — async mid-turn HTTP tool calls with pre-LLM filler gate (<100ms perceived latency)
- `HandoffProtocol` — 3-phase human handoff (decision → bridge → teardown)
- Barge-in handler (`handle_barge_in`) — single event loop tick, audio buffer flush
- Silence handler (`handle_silence`)
- Pipecat integration (`voxmaestro.integrations.pipecat`) — `VoxMaestroPipecatProcessor` frame processor
- Real estate agent example config (`examples/real_estate_agent.yaml`)
- Apache-2.0 license
