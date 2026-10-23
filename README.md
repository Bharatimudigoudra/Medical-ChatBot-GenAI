# MedGuide AI - LangGraph + Pinecone + FastAPI + Groq

An educational medical assistant. Ask a question in text or voice, optionally attach a photo, and get an answer grounded in a medical reference book (your own Pinecone index) with an optional web fallback, plus explicit uncertainty notes and source citations.

This is a tutorial reference implementation for learning. It is not a clinically validated diagnostic system and not a HIPAA-compliant deployment. Safety behavior is instructed through the system prompt, not guaranteed.

## How it works

1. **Retrieve** - the question (plus the photo description, if a photo was attached) is embedded locally with `sentence-transformers/all-MiniLM-L6-v2` and matched against an existing 384-dimensional Pinecone index (`medicalbot`). No re-ingestion is needed; the index is reused as-is. Book excerpts below the relevance score are dropped.
2. **Route** - if the book match is weak or missing, or the question asks for current guidance, the query goes to web search. Otherwise it goes straight to answering.
3. **Web (optional)** - Tavily search runs only when routed to web and `TAVILY_API_KEY` is set.
4. **Answer** - a Groq chat model composes the answer from the question, recent history, book excerpts, web results, and any photo observations. Sources are cited as `[PDF p.N]` (1-based page numbers) or web URLs.

Photo analysis is a separate first step: the vision model produces an objective description of what is visible (no diagnosis, no measurements). If photo analysis fails, the app says so explicitly and does not answer from the photo.

## Models (all via Groq)

| Role | Model |
|------|-------|
| Answer generation | `openai/gpt-oss-120b` |
| Photo description | `qwen/qwen3.8-27b` |
| Voice transcription | `whisper-large-v3-turbo` |
| Query embeddings (local, not an API call) | `sentence-transformers/all-MiniLM-L6-v2` |

## Features

- Chat UI in vanilla JS: Enter sends, Shift+Enter adds a new line.
- Photo upload (PNG/JPEG/WebP up to 5 MB) or camera capture; the photo appears inside your sent message.
- Voice input: record with the microphone (live level meter), transcribed to text on the server.
- Read-aloud: one control in the toolbar. The "Read answer" text reads the latest answer from the start (click again to restart); the speaker icon mutes and unmutes, continuing from the same position. Uses the browser's own speechSynthesis voices.
- Per-phase timing logs in the server console (`TIMING vision/retrieve/route/web/answer/total`) to show where each request spends its time.
- Conversation memory per thread in a local SQLite file (`medical_memory.db`, created on first run, not committed).
- Optional `API_TOKEN` check for the API; CORS restricted to local origins.

## Requirements

- Python 3.11
- A Groq API key and a Pinecone API key
- A Tavily API key (optional; web fallback is skipped without it)
- Internet connection

## Setup

```
conda create -n medicalbot python=3.11 -y
conda activate medicalbot
pip install -r requirements.txt
```

Create a `.env` file in the project folder:

```
GROQ_API_KEY=...
PINECONE_API_KEY=...
TAVILY_API_KEY=...        # optional
# API_TOKEN=...           # optional: require a bearer token for the API
```

## Run

```
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. API documentation: http://127.0.0.1:8000/docs. `GET /api/health` reports whether keys are configured.

## Structure

- `app/config.py` - environment settings and the conda SSL certificate workaround
- `app/knowledge.py` - local MiniLM embedding + Pinecone retrieval from the existing index
- `app/agent.py` - LangGraph flow: retrieve, route, optional web search, answer
- `app/main.py` - FastAPI endpoints: chat, photo analysis, transcription, health
- `templates/index.html`, `static/app.js`, `static/style.css` - the web UI
- `docs/architecture-medguide-groq.png` - architecture diagram of the actual data flow
- `data/` - user-supplied reference PDF (not committed). The bundled Gale Encyclopedia of Medicine (2002, Volume 1 A-B) is a historical source; answers identify it as such
- `backup-*/` - safety copies made by update scripts (not committed)

## Troubleshooting

- **Connecting / checking:** start the server and open `/api/health` first. Check the browser console and the server terminal.
- **401:** if `API_TOKEN` is set, enter it in the UI's "API access" section. Do not expose this token in a public deployment.
- **502:** check the server terminal for a `Chat error:` line; verify Groq/Pinecone keys, the Pinecone index name, and connectivity.
- **Slow answers:** each request prints `TIMING` lines in the server terminal showing how many seconds each phase took (photo analysis, book retrieval, web search, answer generation, total).
- **Photo not analyzed:** the answer will say so explicitly. Retry, or remove the photo and ask as text.
- **Read-aloud:** uses your browser's built-in voices; mute/unmute continues from the same position on voices that report their progress.
