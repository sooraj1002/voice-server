# Expense Tracker Voice Server

Local voice assistant for the expense tracker. Runs on your Fedora GPU machine.

**Stack:** faster-whisper (STT) → Ollama/qwen3:8b (LLM + tool calling) → Kokoro ONNX (TTS)

---

## Setup

### 1. Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:8b
```

Set `OLLAMA_KEEP_ALIVE=-1` so Ollama doesn't auto-evict the model:

```bash
sudo systemctl edit ollama
```
Add under `[Service]`:
```
Environment="OLLAMA_KEEP_ALIVE=-1"
```
Then `sudo systemctl restart ollama`.

### 2. Install CUDA & ffmpeg

```bash
# Fedora
sudo dnf install ffmpeg
# CUDA: follow NVIDIA's guide for your driver version
# NeMo also needs libsndfile
sudo dnf install libsndfile
```

### 3. Download Kokoro ONNX model

```bash
pip install huggingface-hub
huggingface-cli download thewh1teagle/kokoro-onnx kokoro-v1.0.onnx voices-v1.0.bin --local-dir ./models
```

### 4. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Configure

```bash
cp .env.example .env
# Edit .env:
#   BACKEND_URL=http://shadywrldserver:8082
#   KOKORO_ONNX_PATH=./models/kokoro-v1.0.onnx
#   KOKORO_VOICES_PATH=./models/voices-v1.0.bin
#   ALLOWED_ORIGINS=https://your-tailscale-hostname
```

### 6. Run

```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8765
```

### 7. (Optional) systemd service

```ini
# /etc/systemd/system/voice-server.service
[Unit]
Description=Expense Tracker Voice Server
After=network.target ollama.service

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/path/to/voice-server
EnvironmentFile=/path/to/voice-server/.env
ExecStart=/path/to/voice-server/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8765
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## Testing

```bash
# Check status
curl http://localhost:8765/status

# Wake model (replace TOKEN with a valid JWT from the app)
curl -X POST http://localhost:8765/wake -H "Authorization: Bearer TOKEN"

# Test WebSocket (install websocat)
websocat "ws://localhost:8765/ws?token=TOKEN"
# Then type: {"type":"end_turn"}
```

---

## Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| POST   | `/wake`   | Load LLM model into VRAM             |
| GET    | `/status` | Model load state + last activity     |
| WS     | `/ws`     | Voice session (token in query param) |

## WebSocket Protocol

**Client → Server:**
- Binary frames: WebM/Opus audio chunks (sent every 250ms by browser)
- Text `{"type":"end_turn"}`: process buffered audio and respond
- Text `{"type":"cancel"}`: discard buffer

**Server → Client:**
- Text `{"type":"thinking"}`: processing started
- Text `{"type":"transcript","text":"..."}`: what the user said
- Binary: WAV audio (TTS response)
- Text `{"type":"done"}`: ready for next turn
- Text `{"type":"tool_result","tool":"add_expense","success":true}`: expense was added
- Text `{"type":"error","message":"..."}`: something went wrong
