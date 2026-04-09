<p align="center">
  <h1 align="center">🎼 VoxMaestro</h1>
  <p align="center"><strong>The open source voice agent conductor.</strong></p>
  <p align="center">Real-time conversation orchestration for AI voice agents.<br/>One YAML file. Full control. Zero vendor lock-in.</p>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#why-voxmaestro">Why VoxMaestro</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#schema-reference">Schema</a> •
  <a href="#integrations">Integrations</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## The Problem

You can run Whisper for STT. You can run Piper for TTS. You can run any LLM for generation.

**But who conducts the conversation?**

When your voice agent needs to check a calendar mid-sentence, handle a barge-in within 100ms, or transfer to a human with full context — there's no open source layer for that. Twilio, Vapi, Bland AI, and Retell all keep this proprietary.

VoxMaestro is the missing orchestration layer between your audio pipeline and your AI models.

## Why VoxMaestro

| Problem | Before VoxMaestro | With VoxMaestro |
|---|---|---|
| Conversation flow | Hardcoded `if/else` chains | Declarative YAML state machine |
| Mid-call tool use | Dead air while API responds | Pre-LLM filler gate (<100ms perceived) |
| Human handoff | "Please hold" + cold transfer | 3-phase protocol with context payload |
| Barge-in handling | Ignore it or break the pipeline | Single event loop tick, audio buffer flush |
| Vendor lock-in | Rewrite everything to switch | Swap YAML config, keep your models |
| Observability | `print("DEBUG: got here")` | Structured traces via Langfuse/OpenTelemetry |

## Quickstart

```bash
pip install voxmaestro
```

```python
from voxmaestro import VoxMaestro

# Load your agent from a YAML config
conductor = VoxMaestro.from_yaml("my_agent.yaml")

# Start a call
ctx = conductor.new_call(call_id="call-001", caller_phone="+15551234567")

# Process each caller utterance
result = await conductor.process_turn(ctx, "Do you have anything Thursday at 3?")

# result.filler → "Let me check what we have available."  (fires immediately)
# result.tool_result → { "available": true, "slots": ["3:00 PM", "3:30 PM"] }
# result.state → "tool_call" → returns to "qualification"
```

### With Pipecat

```python
from voxmaestro import VoxMaestro
from voxmaestro.integrations.pipecat import VoxMaestroPipecatProcessor

conductor = VoxMaestro.from_yaml("my_agent.yaml")
processor = VoxMaestroPipecatProcessor(conductor=conductor)

# Drop into your Pipecat pipeline:
#   STT → [VoxMaestro] → LLM → TTS
pipeline = Pipeline([
    transport.input(),
    stt,
    processor,      # ← Orchestration lives here
    llm,
    tts,
    transport.output(),
])
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  VoxMaestro                       │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Intent   │  │  State   │  │  Tool Bridge  │  │
│  │Classifier │→ │ Machine  │→ │  (pre-LLM     │  │
│  │ (VSAI)   │  │ (YAML)   │  │   filler gate) │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│        │              │              │            │
│        │              ▼              │            │
│        │       ┌──────────┐         │            │
│        │       │ Handoff  │         │            │
│        │       │ Protocol │         │            │
│        │       │(3-phase) │         │            │
│        │       └──────────┘         │            │
│        │              │              │            │
└────────┼──────────────┼──────────────┼────────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐
    │   STT   │   │  Telephony │  │  LLM +  │
    │(Whisper)│   │  (Twilio)  │  │   TTS   │
    └─────────┘   └───────────┘  └─────────┘
```

### Hot Path vs Cold Path

VoxMaestro separates real-time voice operations from persistent workflow operations:

- **Hot path** (voice loop): Intent classification → state transition → filler/generation. All in-process memory. No serialization. Sub-200ms budget.
- **Cold path** (handoff/logging): Context payload delivery, transcript persistence, training data capture. Async, durable, can tolerate latency.

The barge-in handler is the purest hot path operation — it must cancel TTS output, flush the audio buffer, and resume STT in a single event loop tick. No network calls, no disk I/O.

## Schema Reference

VoxMaestro agents are defined in YAML. Think `docker-compose.yml` for voice agents.

### Minimal Config

```yaml
schema_version: "0.1.0"
agent:
  name: "my-agent"
  voice:
    provider: "piper"
    model: "en_US-amy-medium"

intent:
  provider: "ollama"
  endpoint: "http://localhost:11434/v1/chat/completions"
  model: "my-intent-model"
  intents:
    - id: "greeting"
      description: "Hello, hi"
    - id: "question"
      description: "Asking for information"
    - id: "unknown"
      description: "Can't classify"

generation:
  provider: "ollama"
  endpoint: "http://localhost:11434/v1/chat/completions"
  model: "llama3"
  max_tokens: 150

states:
  initial:
    transitions:
      greeting: "conversation"
      "*": "conversation"
  conversation:
    transitions:
      "*": "conversation"
```

### Full Config Sections

| Section | Purpose |
|---|---|
| `agent` | Name, language, voice provider + fallback chain |
| `intent` | Classifier endpoint, model, intent definitions with tool/trigger mappings |
| `generation` | LLM endpoint, model, system prompt, generation params |
| `states` | State machine — states, transitions, max_turns, scoring, escalation |
| `tools` | Mid-turn tool definitions with endpoints, fillers, timeout, failure handling |
| `handoff` | Delivery channels (webhook, Slack) and context payload fields |
| `guardrails` | Max duration, silence handling, barge-in, PII redaction, profanity |
| `observability` | Logging provider, metrics to track |

See [`examples/real_estate_agent.yaml`](examples/real_estate_agent.yaml) for a complete production config.

## Key Concepts

### Mid-Turn Tool Bridge

When a caller says something that requires an API call (checking availability, looking up data), VoxMaestro:

1. **Plays a filler immediately** — pre-rendered audio or static text, bypasses LLM entirely
2. **Fires the tool call async** — your API, your endpoint, your timeout
3. **Injects the result** — feeds response into LLM context for natural reply
4. **Resumes generation** — caller hears the answer, not dead air

The filler fires in <100ms because it's a pre-LLM gate — the lookup from intent → filler is a dict lookup, not a model call.

### 3-Phase Handoff Protocol

Human handoff isn't a single event. VoxMaestro implements three phases:

1. **Decision** — State machine determines handoff is needed. Logs reason.
2. **Bridge** — Filler plays ("Let me connect you..."). Context payload fires async to webhook/Slack/CRM. Caller hears hold audio.
3. **Teardown** — Telephony transfer completes. Transcript saved. State flushed to training data. Pipeline closes gracefully.

### Guardrails

- **Max call duration** — hard cap prevents runaway calls
- **Silence detection** — prompts caller after configurable silence
- **Barge-in** — cancels TTS, flushes buffer, resumes listening
- **PII redaction** — configurable allow/redact lists for logging
- **Max turns per state** — prevents infinite loops with escalation

## Integrations

| Framework | Status | Module |
|---|---|---|
| [Pipecat](https://github.com/pipecat-ai/pipecat) | ✅ Built-in | `voxmaestro.integrations.pipecat` |
| [LiveKit Agents](https://github.com/livekit/agents) | 🔜 Planned | — |
| [Vocode](https://github.com/vocodedev/vocode-python) | 🔜 Planned | — |
| Custom / Standalone | ✅ Works now | `voxmaestro.conductor` |

### Pluggable Components

VoxMaestro is intentionally **model-agnostic** and **transport-agnostic**:

- **Intent classifier**: Any HTTP endpoint returning an intent string. Ship your own fine-tuned model or use OpenAI.
- **Generation model**: Any OpenAI-compatible endpoint. Local (Ollama, MLX), cloud, hybrid.
- **TTS/STT**: Not VoxMaestro's job. Use whatever your pipeline provides.
- **Telephony**: Twilio, Vonage, SIP — VoxMaestro doesn't care. It orchestrates conversations, not phone calls.

## Development

```bash
git clone https://github.com/gentic-ai/voxmaestro.git
cd voxmaestro
pip install -e ".[dev]"
pytest tests/ -v
```

## Roadmap

- [ ] LiveKit Agents integration
- [ ] Visual state machine editor (web UI)
- [ ] Multi-language conversation support
- [ ] Conversation analytics dashboard
- [ ] n8n node for cold-path workflows
- [ ] State machine hot-reload without call interruption
- [ ] Conversation replay from transcript (testing/debugging)

## Contributing

VoxMaestro is Apache-2.0 licensed. Contributions welcome.

The biggest impact areas right now:
1. **LiveKit integration** — bring the same frame processor pattern to LiveKit Agents
2. **More example configs** — healthcare, customer support, restaurant booking
3. **Visual state machine editor** — render YAML as an interactive graph
4. **Conversation simulator** — test configs without a live phone line

## License

Apache License 2.0 — use it in production, fork it, sell services on top of it.

---

<p align="center">
  <strong>Built by <a href="https://genticai.pro">Gentic AI Solutions</a></strong><br/>
  AI automation infrastructure for businesses that move fast.
</p>
