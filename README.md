# History Agent 2 — Role-play Historical Figures + TTS

A lightweight Google ADK app that chats **in first person** as a named historical figure.  
It fetches a concise factual profile + voice style, composes a reply, and speaks it aloud with **ElevenLabs TTS** (with **Gemini TTS** fallback). Audio is saved locally and served via a tiny HTTP server so the chat UI can play it.

---

## Features

- **Role-play only**: stays strictly in character for the figure you pick.
- **Factual grounding**: `get_details(person_name)` builds a short profile.
- **Style guidance**: `get_voice_style(person_name)` returns first-person style samples.
- **Auto voice picking**: picks a region-appropriate ElevenLabs voice (male/female), else falls back to Gemini TTS.
- **One final message**: instruction ensures the UI gets a single combined text + audio message.
- **Local audio host**: writes `.mp3/.wav` to a temp folder and serves via `http://127.0.0.1:<port>/...`.

---

## Requirements

- **Python** 3.10+ (tested with 3.12)
- **Google ADK** CLI (`adk`)
- Python packages:
  - `google-adk`
  - `google-genai`
  - `python-dotenv`
  - `requests`

Install:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -U pip
pip install google-adk google-genai python-dotenv requests
