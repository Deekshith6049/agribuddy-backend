from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from supabase import create_client
import os
from dotenv import load_dotenv
from langdetect import detect
from groq import Groq
from gtts import gTTS
import uuid

# ================= LOAD ENV =================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ================= CLIENTS =================
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# ================= FASTAPI =================
app = FastAPI(
    title="AGRIBudy AI Assistant Backend",
    version="1.1.0",
    description="AI-powered agriculture advisor with voice output"
)

# ================= AUDIO STORAGE =================
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

class ChatRequest(BaseModel):
    message: str
    voice: bool = False

# ================= UTIL FUNCTIONS =================
def detect_language(text: str) -> str:
    try:
        code = detect(text)
    except:
        code = "en"

    if code == "te":
        return "Telugu"
    elif code == "hi":
        return "Hindi"
    return "English"

def map_tts_language(language: str) -> str:
    # gTTS language codes
    if language == "Hindi":
        return "hi"
    if language == "Telugu":
        return "te"
    return "en"

def get_latest_sensor_data() -> str:
    res = supabase.table("Soil_data") \
        .select("*") \
        .order("monitored_at", desc=True) \
        .limit(1) \
        .execute()

    if not res.data:
        return "No sensor data available."

    d = res.data[0]
    return (
        f"Temperature: {d.get('temperature')}°C, "
        f"Humidity: {d.get('humidity')}%, "
        f"Soil Moisture: {d.get('soil_moisture')}%, "
        f"Pest Detected: {d.get('pest_detected')}."
    )

def generate_voice(text: str, language: str) -> str:
    lang_code = map_tts_language(language)
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    tts = gTTS(text=text, lang=lang_code)
    tts.save(filepath)

    return f"/audio/{filename}"

# ================= AI CHAT =================
@app.post("/api/ai/chat")
def ai_chat(req: ChatRequest):
    language = detect_language(req.message)
    context = get_latest_sensor_data()

    system_prompt = (
        "You are AGRIBudy AI, a farmer-friendly agriculture advisor. "
        "Give short, clear, practical recommendations and warnings."
    )

    user_prompt = (
        f"Live sensor data: {context}\n"
        f"User question: {req.message}\n"
        f"Respond in {language}."
    )

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        max_tokens=300
    )

    reply_text = response.choices[0].message.content.strip()

    audio_url = None
    if req.voice:
        audio_url = generate_voice(reply_text, language)

    return {
        "language": language,
        "reply_text": reply_text,
        "voice_enabled": req.voice,
        "audio_url": audio_url
    }

@app.get("/")
def health():
    return {"status": "AGRIBudy backend running with voice output"}
