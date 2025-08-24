from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import io, wave, uuid
import os, threading, socket, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import tempfile
import requests, hashlib


def get_details(person_name: str):
    load_dotenv() 
    key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=
        f"""You are a factual profiler for historical figures.

                Target person: "{person_name}"

                Your job:
                - Produce a concise, factual profile about the target.
                - Prefer widely accepted facts. If multiple candidates match, choose the most likely and note that in "disambiguation".
                - Keep text snippets short and useful for role-play.

                STRICT OUTPUT RULES
                - Return ONLY a valid Python dictionary literal (not JSON). No backticks, no prose, no prefixes/suffixes.
                - Use single quotes for keys and strings.
                - Use Python types: int, float, bool, None, list, dict.
                - Years must be ints when known; otherwise use None.
                - Limits: each text field ≤ 35 words; each quote ≤ 20 words; each summary ≤ 2 sentences.
                - If unsure about a field, set it to None (or an empty list where appropriate).

                Output EXACTLY the following dictionary (keys in this order):

                {{
                'canonical_name': <str>,
                'aliases': <list[str]>,
                'birth_year': <int|None>,
                'death_year': <int|None>,
                'age_at_death': <int|None>,  
                'nationality': <str|None>,
                'roles': <list[str]>,        
                'fields': <list[str]>,       
                'era': <str|None>,          
                'summary': <str>,           
                'major_breakthroughs': [    
                    {{
                    'title': <str>,
                    'year': <int|None>,
                    'summary': <str>         
                    }}
                ],
                'notable_works': [           
                    {{
                    'title': <str>,
                    'year': <int|None>,
                    'type': <str>            
                    }}
                ],
                'key_quotes': <list[str]>,   
                'speaking_style': <list[str]>,  
                'controversies': <list[str]>,   
                'disambiguation': <str|None>,          
                }}

                Example of the required return *style* (not the schema):
                {{'canonical_name':'Albert Einstein','age_at_death':76}}  # This is only an illustration of Python dict form.

                Return only the dictionary described above, fully populated for "{person_name}".
                """
        
    )
    final_response = response.text.strip()
    return final_response


def get_voice_style(person_name:str):
    load_dotenv() 
    key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    schema = """
    {
    'samples': [  # 4 to 6 first-person samples in distinct styles
        { 'label': <str>, 'purpose': <str>, 'sample': <str> }
    ]
    }
    """.strip()

    prompt = f"""You create first-person writing samples for historical figures.

    Target person: "{person_name}"

    Goal:
    - Provide 4–6 SHORT first-person samples in DISTINCT styles
    - Be historically plausible; avoid modern slang and caricature.
    - Do not copy long quotes verbatim; paraphrase if needed.
    - Include at least one sample labeled 'polite-decline' that models a graceful, in-character refusal
    with a brief context pivot and an inviting follow-up question.

    STRICT OUTPUT RULES
    - Return ONLY a valid Python dictionary literal with a single key 'samples'. No backticks or extra prose.
    - Use single quotes for keys/strings. Use Python types.
    - Each 'sample' must be 70–120 words, written in first person ('I').
    - 'label' names the style (e.g., 'formal-lecture','personal-letter','public-speech','notebook-entry','interview-q&a','maxim').
    - 'purpose' briefly states when to use that style.

    Required shape:

    {schema}

    Return only the dictionary for "{person_name}".
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt)

    final_response = response.text.strip()

    return final_response

def get_voice_accent(person_name: str):
    load_dotenv() 
    key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    prompt = f"""
    You need to generate a single-line TTS delivery instruction for how this person would have spoke, person name: {person_name}.
    You can use the person's nationality, region and era to decide what kind of voice this person could have.
    
    Rules:
    - Output EXACTLY ONE LINE, no quotes, no extra text.
    - Write in English, describing HOW to speak: accent, tone, tempo, pitch, enunciation, pausing, formality, rhetoric cues.
    - Keep it precise and actionable; avoid modern slang unless era-appropriate.
    - Do not mention these inputs or that this is an instruction; just give the delivery line.

    Return only one line like:
    "Speak in gently German-accented English with a measured tempo, slightly lowered pitch, precise enunciation, brief reflective pauses, formal register, and evidence-first phrasing."
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt)

    final_response = response.text.strip()

    return final_response

def _choose_gemini_voice(nationality: str, gender: str) -> str:
    nat = (nationality or "").lower()
    gen = (gender or "").lower()

    # India / South Asia
    if any(k in nat for k in ["india","indian","pakistan","bangladesh","sri lanka","nepal"]):
        return "Puck" if gen in ["male","m","man"] else ("Leda" if gen in ["female","f","woman"] else "Puck")

    # UK / Ireland
    if any(k in nat for k in ["uk","united kingdom","england","britain","wales","scotland","ireland","irish","british"]):
        return "Kore"

    # USA / Canada
    if any(k in nat for k in ["united states","usa","american","canada","canadian"]):
        return "Puck"

    # Western / Central / Eastern Europe, Russia, MENA, East Asia → neutral/continental
    if any(k in nat for k in ["germany","german","france","french","italy","italian","spain","spanish",
                              "poland","czech","hungary","balkan","russia","russian",
                              "egypt","turkey","saudi","iran","arab","japan","korea","china","chinese"]):
        return "Zephyr"

    # Latin America or Sub-Saharan Africa → use Puck as clearer/neutral
    if any(k in nat for k in ["brazil","argentina","mexico","colombia","peru","chile",
                              "nigeria","south africa","kenya","ghana","ethiopia"]):
        return "Puck"

    # Fallback
    return "Kore"

# Tool that picks for the model
def speak_auto(text: str, nationality: str, gender: str) -> dict:
    voice = _choose_gemini_voice(nationality, gender)
    return speak(text=text, voice=voice)

_AUDIO_DIR = Path(tempfile.gettempdir()) / "history_agent_audio"
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
_AUDIO_PORT = 8765  # change if needed



class _AudioHandler(SimpleHTTPRequestHandler):
    # serve from _AUDIO_DIR and set a good content-type for .wav/.mp3
    def translate_path(self, path):
        # serve only from our audio dir
        rel = path.lstrip("/\\")
        return str(_AUDIO_DIR / rel)

    def end_headers(self):
        # allow cross-origin loads (audio tag)
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()



def _port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0



def _start_audio_server():
    global _AUDIO_PORT
    # try 8765, then a few fallbacks
    for p in (_AUDIO_PORT, 8766, 8770, 8888):
        if not _port_in_use(p):
            _AUDIO_PORT = p
            break
    httpd = ThreadingHTTPServer(("127.0.0.1", _AUDIO_PORT), _AudioHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    # tiny wait so it’s listening before first use
    time.sleep(0.15)

# start once when tools.py is imported
try:
    _start_audio_server()
except Exception:
    pass

from google import genai
from google.genai import types



def speak(text: str, voice: str = "Kore") -> dict:
    """
    Gemini TTS (Preview). Input: final user-facing reply text and a prebuilt voice name.
    Output:
      - audio_url: http://127.0.0.1:<port>/<uuid>.wav  (short URL you can use in <audio>)
      - audio_tag: '<audio controls src="http://127.0.0.1:<port>/<uuid>.wav"></audio>'  (ready to paste)
      - voice: the voice name used
    Always insert 'audio_tag' VERBATIM in the final assistant message (do not retype it, no braces).
    """ 
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

    resp = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,  # IMPORTANT: only the reply text you want spoken
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )

    # Convert 24kHz 16-bit mono PCM -> WAV bytes
    pcm = resp.candidates[0].content.parts[0].inline_data.data
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000); wf.writeframes(pcm)
    wav_bytes = buf.getvalue()

    # Save file and return short URL (no base64 in chat)
    fname = f"{uuid.uuid4().hex}.wav"
    fpath = _AUDIO_DIR / fname
    fpath.write_bytes(wav_bytes)

    url = f"http://127.0.0.1:{_AUDIO_PORT}/{fname}"
    return {
        "audio_url": url,
        "audio_tag": f'<audio controls src="{url}"></audio>',
        "voice": voice,
    }


# --- ELEVENLABS VOICE MAP (compact, clearly distinct accent buckets) ---------
# # Each bucket has male/female so you can respect gender when known.

ELEVEN_VOICE_MAP = {
    # South Asia
    "en_in": {  # India/Pakistan/Bangladesh/Sri Lanka/Nepal
        "male":   "CZdRaSQ51p0onta4eec8",
        "female": "kL06KYMvPY56NluIQ72m"
    },

    # UK & Ireland
    "en_gb": {
        "male":   "aaorr6ZHIL88gEexu7dC",
        "female": "jB2lPb5DhAX6l1TLkKXy"
    },

    # North America (US/Canada)
    "en_us": {
        "male":   "sIT4mjQ8vhgwxvsfq5Li",
        "female": "yM93hbw8Qtvdma2wCnJG"
    },

    # Australia / New Zealand
    "en_au": {
        "male":   "sclx1MZrNqboRcmLWoDb",
        "female": "w9rPM8AIZle60Nbpw7nl"
        
    },

    # Chinese-accented English (distinct from JP/KR)
    "en_cn": {
        "male":   "cLCuNe0GeCZkd2MXpQWN",
        "female": "ykMqqjWs4pQdCIvGPn0z"
       
    },

    # East Asia (Japan/Korea) combined for simplicity
    "en_eastasia": {
        "male":   "UznIBkKIQe3ZG2tGydre",
        "female": "B2j2knC2POvVW0XJE6Hi"
     
    },

    # Middle East & North Africa (Arabic/Persian/Turkish/Hebrew-accented English)
    "en_mena": {
        "male":   "uQPOhlzA94sogqmhGLCI",
        "female": "aCChyB4P5WEomwRsOKRh"
        
    },

    # Continental Europe (German/French/Italian/Spanish/Portuguese/Polish/etc.)
    "en_eu_cont": {
        "male":   "PW7PuXtvQ2D7ViYH2zwB",
        "female": "qomNIe05PS2HOqkLJCkG"
    
    },

    # Latin America (Spanish/Portuguese-accented English)
    "en_latam": {
        "male":   "IP2syKL31S2JthzSSfZH",
        "female": "xwH1gVhr2dWKPJkpNQT9"
        
    },

    # Sub-Saharan Africa (generic)
    "en_af": {
        "male": "nw6EIXCsQ89uJMjytYb8",
        "female": "BcpjRWrYhDBHmOnetmBl"
       
    },
}

# --- NATIONALITY/REGION RESOLVER (compact) -----------------------------------
def _region_key_from_nat(nationality: str) -> str:
    """
    Map nationality/place strings to the closest compact accent bucket above.
    Keep this simple on purpose. Falls back to 'en_gb' if no match.
    """
    n = (nationality or "").lower()

    # South Asia
    if any(k in n for k in ["india","indian","pakistan","pakistani","bangladesh","bangladeshi","sri lanka","sri-lanka","lankan","nepal","nepali"]):
        return "en_in"

    # UK & Ireland
    if any(k in n for k in ["england","english","britain","british","uk","united kingdom","wales","scotland","ireland","irish"]):
        return "en_gb"

    # North America
    if any(k in n for k in ["united states","usa","american","canada","canadian"]):
        return "en_us"

    # Australia / New Zealand
    if any(k in n for k in ["australia","australian","new zealand","nz","kiwi","aotearoa"]):
        return "en_au"

    # China
    if any(k in n for k in ["china","chinese","prc"]):
        return "en_cn"

    # East Asia (JP/KR)
    if any(k in n for k in ["japan","japanese","korea","korean"]):
        return "en_eastasia"

    # MENA (Arabic/Persian/Turkish/Hebrew)
    if any(k in n for k in [
        "uae","saudi","arabia","egypt","morocco","algeria","tunisia","iraq","syria","jordan","lebanon","arab",
        "iran","persian","iranian","turkey","turkish","israel","israeli","hebrew"
    ]):
        return "en_mena"

    # Continental Europe (catch-all)
    if any(k in n for k in [
        "germany","german","france","french","italy","italian","spain","spanish","portugal","portuguese",
        "poland","polish","netherlands","dutch","belgium","austria","switzerland","czech","hungary",
        "romania","bulgaria","serbia","croatia","bosnia","albania","slovenia","slovakia","greece",
        "ukraine","belarus","baltic","estonia","latvia","lithuania"
    ]):
        return "en_eu_cont"

    # Latin America
    if any(k in n for k in [
        "mexico","mexican","brazil","brazilian","argentina","chile","colombia","peru","ecuador","bolivia",
        "paraguay","uruguay","venezuela","latin america","latam","hispanic"
    ]):
        return "en_latam"

    # Sub-Saharan Africa (generic)
    if any(k in n for k in [
        "south africa","nigeria","ghana","kenya","uganda","tanzania","ethiopia","rwanda","burundi","somalia",
        "zambia","zimbabwe","botswana","namibia","angola","cameroon","senegal","mali","ivory coast","côte d'ivoire"
    ]):
        return "en_af"

    # Fallback
    return "en_gb"


def _pick_eleven_voice_id(nationality: str, gender: str) -> tuple[str, str]:
    """
    Return (voice_id, region_key) for ElevenLabs, with no 'neutral' fallback.
    Strategy:
      1) If gender is known, pick that exact voice if present.
      2) If gender unknown or missing key, prefer 'male', else 'female'.
      3) Last resort: first value in the mapping (if any).
    """
    region = _region_key_from_nat(nationality)
    bundle = ELEVEN_VOICE_MAP.get(region) or ELEVEN_VOICE_MAP.get("en_gb", {})

    g = (gender or "").strip().lower()
    # 1) exact gender match
    if g in ("male", "man", "m") and "male" in bundle:
        return bundle["male"], region
    if g in ("female", "woman", "f") and "female" in bundle:
        return bundle["female"], region

    # 2) unknown → prefer male, then female
    if "male" in bundle:
        return bundle["male"], region
    if "female" in bundle:
        return bundle["female"], region

    # 3) last resort: any value
    return (next(iter(bundle.values()), ""), region)


def speak_elevenlabs(text: str, voice_id: str, model_id: str = "eleven_multilingual_v2") -> dict:
    """
    Synthesize with ElevenLabs; returns audio_url + audio_tag.
    Requires ELEVENLABS_API_KEY in env. Writes MP3 into _AUDIO_DIR.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key or not voice_id:
        raise RuntimeError("Missing ELEVENLABS_API_KEY or voice_id")

    # Caching to save quota: hash voice+text
    cache_key = hashlib.sha256((voice_id + "|" + text).encode("utf-8")).hexdigest()[:20]
    fname = f"{cache_key}.mp3"
    fpath = _AUDIO_DIR / fname
    if fpath.exists():
        url = f"http://127.0.0.1:{_AUDIO_PORT}/{fname}"
        return {"audio_url": url, "audio_tag": f'<audio controls src="{url}"></audio>', "voice": voice_id, "engine": "elevenlabs(cache)"}

    url_endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        # Optional fine-tuning per voice:
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    r = requests.post(url_endpoint, headers=headers, json=payload, timeout=60)
    print(f"[TTS] Eleven HTTP {r.status_code} for voice_id={voice_id}")
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs error {r.status_code}: {r.text[:200]}")

    fpath.write_bytes(r.content)
    url = f"http://127.0.0.1:{_AUDIO_PORT}/{fname}"
    return {"audio_url": url, "audio_tag": f'<audio controls src="{url}"></audio>', "voice": voice_id, "engine": "elevenlabs"}

def speak_elevenlabs_auto(text: str, nationality: str, gender: str) -> dict:
    """
    Try ElevenLabs first; on any error OR missing mapping, fall back to Gemini TTS (speak()).
    """
    try:
        voice_id, region = _pick_eleven_voice_id(nationality, gender)
        # If we couldn't map a usable voice, bail to fallback immediately
        if not voice_id:
            raise RuntimeError("No usable ElevenLabs voice_id for this account/region/gender.")

        # Happy path: synth with ElevenLabs
        return speak_elevenlabs(text=text, voice_id=voice_id)

    except Exception as e:
        # Clean fallback to Gemini
        try:
            res = speak(text=text, voice="Kore")
            res["engine"] = "gemini-fallback"
            res["note"] = f"[ElevenLabs fallback: {e}]"
            return res
        except Exception as e2:
            return {
                "audio_url": "",
                "audio_tag": "",
                "voice": "",
                "engine": "none",
                "error": f"[ElevenLabs fallback: {e}]; Gemini failed: {e2}"
            }
