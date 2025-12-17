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

# === NEW IMPORTS (SAFE) ===
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
    version="1.3.0",
    description="AI-powered agriculture advisor with camera-based pest severity & multilingual voice support"
)

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= AUDIO STORAGE =================
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# ================= MODELS =================
class ChatRequest(BaseModel):
    message: str
    voice: bool = False

# ================= LANGUAGE UTILS =================
def detect_language(text: str) -> str:
    try:
        code = detect(text)
    except:
        code = "en"

    return {"te": "Telugu", "hi": "Hindi"}.get(code, "English")

def tts_language(language: str) -> str:
    return {"English": "en", "Hindi": "hi", "Telugu": "te"}.get(language, "en")

# ================= SENSOR DATA =================
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

# ================= CAMERA IMAGE =================
def get_latest_pest_image_url():
    res = supabase.table("Soil_data") \
        .select("pest_image_url") \
        .order("monitored_at", desc=True) \
        .limit(1) \
        .execute()

    if not res.data or not res.data[0].get("pest_image_url"):
        return None

    return res.data[0]["pest_image_url"]

# ================= PEST SEVERITY ANALYSIS =================
def analyze_pest_severity(image_url: str) -> str:
    try:
        response = requests.get(image_url, timeout=6)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_np = np.array(img)

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        yellow = cv2.inRange(hsv, (18, 50, 50), (35, 255, 255))
        brown = cv2.inRange(hsv, (5, 60, 30), (20, 255, 200))

        total = img_np.shape[0] * img_np.shape[1]

        green_ratio = np.count_nonzero(green) / total
        yellow_ratio = np.count_nonzero(yellow) / total
        brown_ratio = np.count_nonzero(brown) / total

        # === Tuned thresholds (field-safe) ===
        if brown_ratio > 0.06:
            return "HIGH"
        elif yellow_ratio > 0.10:
            return "MEDIUM"
        elif green_ratio < 0.55:
            return "MEDIUM"
        else:
            return "LOW"

    except:
        return "UNKNOWN"

# ================= TELUGU ENRICHMENT =================
def enrich_telugu(text: str) -> str:
    if len(text) < 200:
        text += (
            "\n\n👉 రైతులకు సూచన:\n"
            "ఈ సమస్యను నిర్లక్ష్యం చేయకుండా వెంటనే చర్యలు తీసుకుంటే "
            "పంట నష్టం తగ్గించవచ్చు. క్రమం తప్పకుండా పొలాన్ని పరిశీలించడం చాలా అవసరం."
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
    sensor_context = get_latest_sensor_data()

    image_url = get_latest_pest_image_url()
    pest_severity = "UNKNOWN"
    if image_url:
        pest_severity = analyze_pest_severity(image_url)

    system_prompt = (
        "You are AGRIBudy AI, a farmer-friendly agriculture advisor. "
        "Give practical, field-ready advice. Be clear and calm."
    )

    user_prompt = (
        f"Live sensor data: {sensor_context}\n"
        f"Pest risk level based on plant image: {pest_severity}\n"
        f"User question: {req.message}\n"
        f"Respond in {language} with actionable steps."
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

    reply_text = response.choices[0].message.content.strip()

    if language == "Telugu":
        reply_text = enrich_telugu(reply_text)

    audio_url = generate_audio(reply_text, language) if req.voice else None

    return {
        "language": language,
        "reply_text": reply_text,
        "voice_enabled": req.voice,
        "audio_url": audio_url,
        "pest_severity": pest_severity
    }

@app.get("/")
def health():
    return {"status": "AGRIBudy backend running (camera + AI enabled)"}
