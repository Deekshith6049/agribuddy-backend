from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from supabase import create_client
import os
from dotenv import load_dotenv
from langdetect import detect
from groq import Groq
from gtts import gTTS
import uuid

# === IMAGE / CV ===
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

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
    version="1.4.0",
    description="Camera-based pest severity + real-time soil intelligence"
)

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= AUDIO =================
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# ================= MODELS =================
class ChatRequest(BaseModel):
    message: str
    voice: bool = False

# ================= LANGUAGE =================
def detect_language(text: str) -> str:
    try:
        code = detect(text)
    except:
        return "English"
    return {"te": "Telugu", "hi": "Hindi"}.get(code, "English")

def tts_language(language: str) -> str:
    return {"English": "en", "Hindi": "hi", "Telugu": "te"}.get(language, "en")

# ================= SENSOR FETCH (COLUMN-BASED) =================
def get_latest_non_null(column: str):
    res = supabase.table("Soil_data") \
        .select(column) \
        .not_.is_(column, None) \
        .order("monitored_at", desc=True) \
        .limit(1) \
        .execute()
    return res.data[0][column] if res.data else None

def get_latest_sensor_data():
    return {
        "temperature": get_latest_non_null("temperature"),
        "humidity": get_latest_non_null("humidity"),
        "soil_moisture": get_latest_non_null("soil_moisture"),
        "soil_ph": get_latest_non_null("soil_ph"),
        "nitrogen": get_latest_non_null("nitrogen"),
        "phosphorus": get_latest_non_null("phosphorus"),
        "potassium": get_latest_non_null("potassium"),
        "pest_detected": get_latest_non_null("pest_detected")
    }

# ================= IMAGE FETCH (FIXED) =================
def get_latest_pest_image_url():
    res = supabase.table("Soil_data") \
        .select("pest_image_url") \
        .not_.is_("pest_image_url", None) \
        .order("monitored_at", desc=True) \
        .limit(1) \
        .execute()

    if not res.data:
        return None

    return res.data[0]["pest_image_url"]

# ================= PEST SEVERITY (HIGH SENSITIVITY) =================
def analyze_pest_severity(image_url: str) -> str:
    try:
        response = requests.get(image_url, timeout=6)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        green = cv2.inRange(hsv, (30, 30, 30), (90, 255, 255))
        yellow = cv2.inRange(hsv, (15, 40, 40), (35, 255, 255))
        brown = cv2.inRange(hsv, (0, 40, 20), (20, 255, 200))
        dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))

        total = img_np.shape[0] * img_np.shape[1]

        g = np.count_nonzero(green) / total
        y = np.count_nonzero(yellow) / total
        b = np.count_nonzero(brown) / total
        d = np.count_nonzero(dark) / total

        # 🔥 VERY SENSITIVE THRESHOLDS
        if d > 0.80 or b > 0.80:
            return "HIGH"
        elif y > 0.10 or g < 0.60:
            return "MEDIUM"
        elif g > 0.10:
            return "LOW"
        else:
            return "LOW"

    except:
        return "UNKNOWN"

# ================= TELUGU ENRICHMENT =================
def enrich_telugu(text: str) -> str:
    if len(text) < 220:
        text += (
            "\n\n👉 రైతులకు సూచన:\n"
            "ఈ పరిస్థితిని గమనించి ముందస్తు చర్యలు తీసుకుంటే "
            "పంట ఆరోగ్యం మెరుగుపడుతుంది."
        )
    return text

# ================= AUDIO =================
def generate_audio(text: str, language: str) -> str:
    filename = f"{uuid.uuid4()}.mp3"
    path = os.path.join(AUDIO_DIR, filename)
    gTTS(text=text, lang=tts_language(language)).save(path)
    return f"/audio/{filename}"

# ================= AI CHAT =================
@app.post("/api/ai/chat")
def ai_chat(req: ChatRequest):
    language = detect_language(req.message)
    sensors = get_latest_sensor_data()

    image_url = get_latest_pest_image_url()
    pest_severity = analyze_pest_severity(image_url) if image_url else "UNKNOWN"

    system_prompt = (
        "You are AGRIBudy AI, a practical agriculture advisor. "
        "Provide clear, actionable farming guidance."
    )

    user_prompt = (
        f"Sensor Data: {sensors}\n"
        f"Pest Risk Level: {pest_severity}\n"
        f"User Question: {req.message}\n"
        f"Respond in {language}."
    )

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
        max_tokens=350
    )

    reply = response.choices[0].message.content.strip()

    if language == "Telugu":
        reply = enrich_telugu(reply)

    audio_url = generate_audio(reply, language) if req.voice else None

    return {
        "language": language,
        "reply_text": reply,
        "voice_enabled": req.voice,
        "audio_url": audio_url,
        "pest_severity": pest_severity,
        "image_used": image_url
    }

# ================= DIRECT API =================
@app.get("/api/pest-severity")
def pest_severity():
    image_url = get_latest_pest_image_url()
    if not image_url:
        return {"pest_severity": "UNKNOWN"}
    return {
        "pest_severity": analyze_pest_severity(image_url),
        "image_used": image_url
    }

@app.get("/")
def health():
    return {"status": "AGRIBudy backend running successfully 🚀"}
