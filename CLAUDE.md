# Expense Tracker — Voice Server

## Purpose
Local voice assistant that runs on the Fedora GPU machine. Accepts audio from the browser via WebSocket, transcribes it, passes it to a local LLM with tool calling, and speaks the response back. Calls the expense tracker backend REST API to add/query expenses on behalf of the user.

## Stack
- **Python** + **FastAPI** + **WebSockets**
- **Parakeet TDT 1.1b** (NVIDIA NeMo) — STT, English only, CUDA
- **qwen3:8b** via **Ollama** — LLM with tool calling
- **Kokoro ONNX** (82M) — TTS, falls back to Piper if model files missing
- **ffmpeg** — WebM/Opus → PCM conversion (must be installed on system)

## Running
```bash
# First time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy and fill in config
cp .env.example .env

# Run
uvicorn main:app --host 0.0.0.0 --port 8765

# Or directly
python main.py
```

Ollama must be running separately (`sudo systemctl start ollama`). See README.md for full Fedora setup.

## File structure
```
main.py           — FastAPI app, endpoints, WebSocket handler
pipeline.py       — Orchestrates STT → LLM → tools → TTS for one voice turn
stt.py            — Parakeet TDT 1.1b via NeMo
tts.py            — Kokoro ONNX (or Piper fallback)
llm.py            — Ollama /api/chat client, tool call parsing
tools.py          — Tool schemas + implementations (calls backend REST API)
model_manager.py  — Wake/sleep lifecycle for the Ollama model
```

## WebSocket protocol
**Client → Server:**
- Binary frames: WebM/Opus audio chunks (250ms each from MediaRecorder)
- Text `{"type": "end_turn"}` — process buffered audio
- Text `{"type": "cancel"}` — discard buffer
- Text `{"type": "confirm_result", "confirmed": bool}` — reply to confirm_expense

**Server → Client:**
- Text `{"type": "thinking"}` — processing started
- Text `{"type": "transcript", "text": "..."}` — what user said
- Text `{"type": "confirm_expense", "details": {...}}` — awaits confirm_result before posting expense
- Binary: WAV audio bytes (TTS response)
- Text `{"type": "done"}` — ready for next turn
- Text `{"type": "tool_result", "tool": "add_expense", "success": true}` — triggers React Query invalidation in browser
- Text `{"type": "error", "message": "..."}` — something failed

## Confirmation flow
`add_expense` always goes through a confirmation step:
1. Server calls `prepare_add_expense()` — resolves category/account IDs via fuzzy match, builds display dict
2. Sends `confirm_expense` to browser with human-readable details (amount, description, category, account, date)
3. Waits for `confirm_result` from browser
4. If confirmed: calls `execute_prepared_expense()` — POSTs to backend
5. If cancelled: tells LLM "User cancelled" so it responds naturally

## Tools available to the LLM
- `add_expense(amount, description, category_name, account_name, date, tags)` — always requires confirmation
- `list_categories()` — returns user's categories
- `list_accounts()` — returns accounts with balances
- `get_spending_summary(period)` — totals for today/week/month/year

Category and account names are fuzzy-matched (case-insensitive substring). If no match, defaults to first in list.

## Model lifecycle
- Ollama runs as a systemd service with `OLLAMA_KEEP_ALIVE=-1`
- `ModelManager` loads the model on demand when `POST /wake` is called
- Model stays in VRAM as long as there's activity
- Watchdog unloads after `INACTIVITY_TIMEOUT` seconds (default 600) of no voice turns
- `model_manager.touch()` is called after every completed pipeline turn

## Environment variables
```
BACKEND_URL=http://shadywrldserver:8082
OLLAMA_URL=http://localhost:11434
VOICE_MODEL=qwen3:8b
PARAKEET_MODEL=nvidia/parakeet-tdt-1.1b
KOKORO_ONNX_PATH=./models/kokoro-v1.0.onnx
KOKORO_VOICES_PATH=./models/voices-v1.0.bin
KOKORO_VOICE=af_heart
INACTIVITY_TIMEOUT=600
ALLOWED_ORIGINS=https://<tailscale-frontend-hostname>
HOST=0.0.0.0
PORT=8765
```

## Auth
- Browser passes JWT as WebSocket query param: `ws://host:8765/ws?token=<jwt>`
- `POST /wake` requires `Authorization: Bearer <token>`
- Token is passed through to all backend API calls — server never validates the JWT itself, the backend does
- Token comes from the `tracker_access` cookie in the browser

## Key decisions
- **No proxy**: browser connects directly to Fedora via Tailscale — adding a proxy through shadywrldserver would add latency for binary audio
- **Buffer full turn**: audio chunks are buffered until `end_turn` is received, then decoded and transcribed as one unit — more accurate than streaming partial recognition
- **Single WAV frame back**: TTS synthesizes the full response and sends it as one binary WebSocket frame — simple, and Kokoro is fast enough (~200ms) that chunking isn't needed
- **Parakeet over Whisper**: English-only but faster and more accurate for this use case; requires NVIDIA GPU + NeMo
