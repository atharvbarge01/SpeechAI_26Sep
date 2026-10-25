import os
import shutil
import uuid
import logging
import re
import difflib
import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from backend/.env or root .env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
import jiwer

from audio_utils import convert_to_wav
import httpx
from fastapi.responses import StreamingResponse, Response
import wave
import numpy as np

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SpeechAI_Backend")

app = FastAPI(title="SpeechAI API", description="AI-Based Speech Therapy & Fluency System API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module 2 Integration (Video & Communication Analysis)
from video_module.router import router as video_router, init_router as init_video_router
app.include_router(video_router)

# Temporary directory for file processing
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# MongoDB Setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
try:
    mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client["speechai_db"]
    mongo_client.server_info()
    logger.info("Successfully connected to MongoDB.")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    db = None

# Whisper Model Initialization (Lazy Loading)
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        import whisper

        logger.info("Loading Whisper 'base' model...")
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully.")
    return whisper_model

# Initialize Module 2 Router with database and whisper loader
init_video_router(db, get_whisper_model)

# ==========================================
# GLOBAL CONFIGURATION CONSTANTS
# ==========================================
WEIGHT_FILLER = 0.20
WEIGHT_STAMMER = 0.25
WEIGHT_PAUSE = 0.20
WEIGHT_RATE = 0.20
WEIGHT_CLARITY = 0.15

COEFF_FILLER = 40.0
COEFF_STAMMER = 50.0
COEFF_PAUSE = 60.0

PAUSE_THRESHOLD_SEC = 1.5
IDEAL_WPM_MIN = 120.0
IDEAL_WPM_MAX = 150.0

WHISPER_FILLER_PROMPT = "Um, uh, umm, uhh, er, ah, hmm, like, you know, actually, basically, so, well, i mean."
SECRET_KEY = "speechai_cryptographic_secret_key_salt_token"
# ==========================================

# ==========================================
# CRYPTOGRAPHIC AUTHENTICATION HELPERS
# ==========================================
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000
    ).hex()
    return f"{salt}:{pwd_hash}"

def verify_password(password: str, stored_password: str) -> bool:
    try:
        salt, pwd_hash = stored_password.split(":")
        test_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()
        return secrets.compare_digest(pwd_hash, test_hash)
    except Exception:
        return False

def generate_token(user_id: str) -> str:
    timestamp = str(int(datetime.utcnow().timestamp()))
    message = f"{user_id}:{timestamp}"
    sig = hmac.new(SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
    return f"{user_id}:{timestamp}:{sig}"

def verify_token(token: str) -> Optional[str]:
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        user_id, timestamp, sig = parts[0], parts[1], parts[2]
        
        token_time = float(timestamp)
        current_time = datetime.utcnow().timestamp()
        # Token valid for 7 days
        if current_time - token_time > 7 * 86400:
            return None
            
        message = f"{user_id}:{timestamp}"
        expected_sig = hmac.new(SECRET_KEY.encode('utf-8'), message.encode('utf-8'), hashlib.sha256).hexdigest()
        if secrets.compare_digest(sig, expected_sig):
            return user_id
        return None
    except Exception:
        return None

def get_user_id_from_header(authorization: Optional[str]) -> str:
    if not authorization:
        return "user_anonymous"
    try:
        if authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            user_id = verify_token(token)
            if user_id:
                return user_id
    except Exception:
        pass
    return "user_anonymous"

# ==========================================
# PYDANTIC SCHEMAS
# ==========================================
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

# ==========================================
# CORE UTILITY FUNCTIONS
# ==========================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def serialize_datetime(dt: datetime) -> str:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')
    return dt.isoformat().replace('+00:00', 'Z')

def detect_acoustic_filled_pauses(wav_path: Optional[str], words_list: Optional[list]) -> List[Dict[str, Any]]:
    """
    Analyzes inter-word intervals in the 16kHz WAV audio.
    If an interval (0.35s to 2.5s) contains sustained vocal energy,
    it is detected as a filled pause (vocal hesitation 'uh/um').
    """
    if not wav_path or not os.path.exists(wav_path) or not words_list or len(words_list) < 2:
        return []

    try:
        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)

        if sampwidth == 2:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        elif sampwidth == 1:
            audio_data = (np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) - 128) * 256
        else:
            return []

        if n_channels > 1:
            audio_data = audio_data[::n_channels]

        total_sec = len(audio_data) / framerate
        if total_sec <= 0:
            return []

        # Estimate noise floor (RMS of quietest 15% of 50ms frames)
        frame_len = int(framerate * 0.05)
        if len(audio_data) < frame_len:
            return []

        frames_rms = []
        for i in range(0, len(audio_data) - frame_len, frame_len):
            chunk = audio_data[i:i+frame_len]
            rms = np.sqrt(np.mean(chunk**2))
            frames_rms.append(rms)

        frames_rms.sort()
        idx_15 = max(1, int(len(frames_rms) * 0.15))
        noise_floor = np.median(frames_rms[:idx_15]) if frames_rms else 50.0
        noise_floor = max(30.0, float(noise_floor))

        vocal_threshold = max(250.0, noise_floor * 2.8)

        filled_pauses = []
        for i in range(len(words_list) - 1):
            curr_end = words_list[i].get("end", 0.0)
            next_start = words_list[i+1].get("start", 0.0)
            gap = next_start - curr_end

            if 0.35 <= gap <= 2.5:
                start_sample = int(curr_end * framerate)
                end_sample = int(next_start * framerate)
                pad = int(0.05 * framerate)
                s_sample = start_sample + pad
                e_sample = end_sample - pad

                if e_sample > s_sample:
                    gap_audio = audio_data[s_sample:e_sample]
                    gap_rms = float(np.sqrt(np.mean(gap_audio**2)))
                    zero_crossings = float(np.sum(np.diff(gap_audio > 0) != 0) / len(gap_audio)) if len(gap_audio) > 0 else 1.0

                    if gap_rms > vocal_threshold and zero_crossings < 0.30:
                        filled_pauses.append({
                            "type": "vocal_hesitation",
                            "start": round(float(curr_end), 2),
                            "end": round(float(next_start), 2),
                            "duration": round(float(gap), 2)
                        })
        return filled_pauses
    except Exception as e:
        logger.error(f"Error in acoustic filled pause detection: {e}")
        return []

def detect_fillers(
    transcript: str, 
    wav_path: Optional[str] = None, 
    words_list: Optional[list] = None,
    return_details: bool = False
):
    text_lower = transcript.lower()
    filler_count = 0
    fillers_found = []
    filler_details = []

    # 1. Multi-word phrase fillers
    phrase_fillers = [
        "you know", "i mean", "sort of", "kind of", 
        "you see", "as in"
    ]
    remaining_text = text_lower
    for phrase in phrase_fillers:
        pattern = rf'\b{re.escape(phrase)}\b'
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            filler_count += len(matches)
            fillers_found.append(phrase)
            remaining_text = re.sub(pattern, ' ', remaining_text)

            # If words_list with timestamps is provided, match word positions
            if words_list:
                p_parts = phrase.split()
                p_len = len(p_parts)
                for i in range(len(words_list) - p_len + 1):
                    sub_words = [re.sub(r'[^\w]', '', words_list[i + k].get("word", "")).lower() for k in range(p_len)]
                    if sub_words == p_parts:
                        s_time = round(float(words_list[i].get("start", 0.0)), 1)
                        e_time = round(float(words_list[i + p_len - 1].get("end", 0.0)), 1)
                        dur = round(max(0.1, e_time - s_time), 1)
                        filler_details.append({
                            "word": phrase,
                            "start": s_time,
                            "end": e_time,
                            "duration": dur
                        })

    # 2. Single-word fillers and hesitations
    cleaned_remaining = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"]', ' ', remaining_text)
    words = cleaned_remaining.split()

    single_word_fillers = {
        "like", "actually", "basically", "literally", "honestly", "seriously", "so", "well"
    }

    for w in words:
        if w in single_word_fillers:
            filler_count += 1
            if w not in fillers_found:
                fillers_found.append(w)
        # Match phonetic vocal hesitations: um, umm, ummm, uh, uhh, uhhh, er, err, erm, ah, ahh, hmm
        elif re.match(r'^u+m+h*$', w):
            filler_count += 1
            if "um" not in fillers_found:
                fillers_found.append("um")
        elif re.match(r'^u+h+m*$', w):
            filler_count += 1
            if "uh" not in fillers_found:
                fillers_found.append("uh")
        elif re.match(r'^e+r+m*$', w):
            filler_count += 1
            if "er/erm" not in fillers_found:
                fillers_found.append("er/erm")
        elif re.match(r'^a+h+$', w):
            filler_count += 1
            if "ah" not in fillers_found:
                fillers_found.append("ah")
        elif re.match(r'^h+m+$', w):
            filler_count += 1
            if "hmm" not in fillers_found:
                fillers_found.append("hmm")

    # Locate single-word timestamps in words_list
    if words_list:
        for w_obj in words_list:
            raw_w = w_obj.get("word", "")
            cw = re.sub(r'[^\w]', '', raw_w).lower()
            if not cw:
                continue
            is_filler = False
            tag_word = cw
            if cw in single_word_fillers:
                is_filler = True
                tag_word = cw
            elif re.match(r'^u+m+h*$', cw):
                is_filler = True
                tag_word = "um"
            elif re.match(r'^u+h+m*$', cw):
                is_filler = True
                tag_word = "uh"
            elif re.match(r'^e+r+m*$', cw):
                is_filler = True
                tag_word = "er/erm"
            elif re.match(r'^a+h+$', cw):
                is_filler = True
                tag_word = "ah"
            elif re.match(r'^h+m+$', cw):
                is_filler = True
                tag_word = "hmm"
            
            if is_filler:
                s_time = round(float(w_obj.get("start", 0.0)), 1)
                e_time = round(float(w_obj.get("end", 0.0)), 1)
                dur = round(max(0.1, e_time - s_time), 1)
                if not any(f["start"] == s_time and f["word"] == tag_word for f in filler_details):
                    filler_details.append({
                        "word": tag_word,
                        "start": s_time,
                        "end": e_time,
                        "duration": dur
                    })

    # 3. Acoustic filled-pause detection (inter-word gap analysis)
    if wav_path and words_list:
        acoustic_fillers = detect_acoustic_filled_pauses(wav_path, words_list)
        if acoustic_fillers:
            filler_count += len(acoustic_fillers)
            if "vocal hesitation (uh/um)" not in fillers_found:
                fillers_found.append("vocal hesitation (uh/um)")
            for af in acoustic_fillers:
                filler_details.append({
                    "word": "vocal hesitation (uh/um)",
                    "start": af["start"],
                    "end": af["end"],
                    "duration": af["duration"]
                })

    filler_details.sort(key=lambda x: x["start"])

    if return_details:
        return filler_count, fillers_found, filler_details
    return filler_count, fillers_found

def detect_stammering(
    transcript: str,
    words_list: Optional[list] = None,
    return_details: bool = False
):
    text_lower = transcript.lower()
    stammer_events = 0
    stammer_details = []

    # 1. Syllable prefix repetitions
    reps_3 = re.findall(r'\b([a-zA-Z]{1,3})-\1-\1\b', text_lower)
    cleaned_text_reps_3_removed = re.sub(r'\b([a-zA-Z]{1,3})-\1-\1\b', ' ', text_lower)
    reps_2 = re.findall(r'\b([a-zA-Z]{1,3})-\1\b', cleaned_text_reps_3_removed)
    
    stammer_events += len(reps_3) + len(reps_2)

    # 2. Consecutive word repetitions
    cleaned_words = clean_text(text_lower).split()
    word_repetitions = 0
    for i in range(len(cleaned_words) - 1):
        if cleaned_words[i] == cleaned_words[i+1]:
            word_repetitions += 1

    stammer_events += word_repetitions

    # If words_list with timestamps is available, locate occurrences
    if words_list:
        for i in range(len(words_list) - 1):
            w1 = re.sub(r'[^\w]', '', words_list[i].get("word", "")).lower()
            w2 = re.sub(r'[^\w]', '', words_list[i+1].get("word", "")).lower()
            if w1 and w1 == w2:
                s_time = round(float(words_list[i].get("start", 0.0)), 1)
                e_time = round(float(words_list[i+1].get("end", 0.0)), 1)
                dur = round(max(0.2, e_time - s_time), 1)
                stammer_details.append({
                    "text": f"{w1} {w2}",
                    "type": "word repetition",
                    "start": s_time,
                    "end": e_time,
                    "duration": dur
                })

        for w_obj in words_list:
            raw_w = w_obj.get("word", "").strip().lower()
            if re.search(r'\b[a-zA-Z]{1,3}-[a-zA-Z]{1,3}', raw_w):
                s_time = round(float(w_obj.get("start", 0.0)), 1)
                e_time = round(float(w_obj.get("end", 0.0)), 1)
                dur = round(max(0.2, e_time - s_time), 1)
                stammer_details.append({
                    "text": raw_w,
                    "type": "syllable prolongation",
                    "start": s_time,
                    "end": e_time,
                    "duration": dur
                })

    stammer_details.sort(key=lambda x: x["start"])

    if return_details:
        return stammer_events, stammer_details
    return stammer_events

def evaluate_condition(condition: str, context: dict) -> bool:
    try:
        sub_conditions = [s.strip() for s in condition.split('and')]
        for sub_cond in sub_conditions:
            parts = sub_cond.split()
            if len(parts) == 3:
                var_name, op, val_str = parts[0], parts[1], parts[2]
                if var_name in context:
                    val = float(val_str)
                    actual_val = context[var_name]
                    
                    if op == "<":
                        if not (actual_val < val): return False
                    elif op == ">":
                        if not (actual_val > val): return False
                    elif op == "<=":
                        if not (actual_val <= val): return False
                    elif op == ">=":
                        if not (actual_val >= val): return False
                    elif op == "==":
                        if not (actual_val == val): return False
                    else:
                        return False
                else:
                    return False
            else:
                return False
        return True
    except Exception as e:
        logger.error(f"Error evaluating condition '{condition}': {e}")
        return False

def seed_exercises():
    if db is None:
        return
    exercises_collection = db["exercises"]
    
    if exercises_collection.count_documents({"feedback_message": {"$exists": True}}) == 0:
        logger.info("Exercises missing feedback messages. Migrating exercises...")
        exercises_collection.drop()
        
        default_exercises = [
            {
                "_id": "silent_pause_drill",
                "title": "Silent Pause Drill",
                "trigger_condition": "filler_score < 6.0",
                "feedback_message": "You used filler words frequently (um, uh, like, you know). Try pausing silently instead of filling gaps.",
                "description": "Read 5 sentences aloud. Replace every urge to say 'um' or 'uh' with a 1-second silent pause.",
                "difficulty": "beginner"
            },
            {
                "_id": "slow_rate_reading",
                "title": "Slow-Rate Reading",
                "trigger_condition": "stammer_score < 6.0",
                "feedback_message": "Repetition of sounds/syllables was detected — a common stammering pattern.",
                "description": "Read a paragraph at half your normal speed, elongating the first sound of each word.",
                "difficulty": "beginner"
            },
            {
                "_id": "sentence_chunking",
                "title": "Sentence-Chunking Practice",
                "trigger_condition": "pause_score < 6.0",
                "feedback_message": "Frequent long hesitations were detected between words.",
                "description": "Break sentences into 3-4 word phrases and practice delivering them phrase-by-phrase with brief pauses.",
                "difficulty": "intermediate"
            },
            {
                "_id": "metronome_paced_reading",
                "title": "Metronome-Paced Reading",
                "trigger_condition": "wpm > 150.0",
                "feedback_message": "Your speech rate is faster than typical conversational pace, which can worsen clarity.",
                "description": "Practice reading a passage to the beat of a metronome set at 120-130 WPM.",
                "difficulty": "intermediate"
            },
            {
                "_id": "timed_reading_challenge",
                "title": "Timed Reading Challenge",
                "trigger_condition": "wpm < 120.0",
                "feedback_message": "Your speech rate is slower than typical — this may indicate hesitation or over-caution.",
                "description": "Read a 150-word passage and try to finish it within exactly 60-70 seconds.",
                "difficulty": "intermediate"
            },
            {
                "_id": "articulation_drill",
                "title": "Articulation Drill",
                "trigger_condition": "clarity_score < 6.0",
                "feedback_message": "Some words were unclear or mumbled. Try articulating consonants more firmly.",
                "description": "Over-enunciate consonants in a set of selected tongue-twisters for 5 minutes daily.",
                "difficulty": "beginner"
            },
            {
                "_id": "advanced_impromptu_speaking",
                "title": "Advanced Impromptu Speaking",
                "trigger_condition": "filler_score >= 6.0 and stammer_score >= 6.0 and pause_score >= 6.0 and clarity_score >= 6.0 and wpm >= 120.0 and wpm <= 150.0",
                "feedback_message": "Excellent fluency! Your pace, pause usage, and articulation are in the optimal range.",
                "description": "Speak impromptu on a random topic for 2 minutes maintaining this excellent flow.",
                "difficulty": "advanced"
            }
        ]
        exercises_collection.insert_many(default_exercises)
        logger.info("Successfully seeded and migrated exercises.")
    else:
        logger.info("Exercises exist and match schema. Skipping seed.")

def seed_default_user():
    if db is None:
        return
    users_collection = db["users"]
    if users_collection.count_documents({"_id": "user_anonymous"}) == 0:
        logger.info("Initializing default anonymous user...")
        users_collection.insert_one({
            "_id": "user_anonymous",
            "name": "Guest User",
            "email": "anonymous@speechai.com",
            "streak_count": 0,
            "last_practice_date": None,
            "created_at": datetime.utcnow()
        })
        logger.info("Default anonymous user initialized.")

def update_user_streak(user_id: str) -> int:
    if db is None:
        return 0
    
    users_collection = db["users"]
    user = users_collection.find_one({"_id": user_id})
    if not user:
        users_collection.insert_one({
            "_id": user_id,
            "name": "Registered User",
            "email": f"{user_id}@speechai.com",
            "streak_count": 1,
            "last_practice_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "created_at": datetime.utcnow()
        })
        return 1

    current_date_str = datetime.utcnow().strftime("%Y-%m-%d")
    last_practice_str = user.get("last_practice_date")
    streak_count = user.get("streak_count", 0)

    if not last_practice_str:
        streak_count = 1
    elif last_practice_str == current_date_str:
        pass
    else:
        last_date = datetime.strptime(last_practice_str, "%Y-%m-%d")
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        delta = (current_date - last_date).days

        if delta == 1:
            streak_count += 1
        else:
            streak_count = 1

    users_collection.update_one(
        {"_id": user_id},
        {
            "$set": {
                "streak_count": streak_count,
                "last_practice_date": current_date_str
            }
        }
    )
    return streak_count

@app.on_event("startup")
def startup_db_client():
    try:
        seed_exercises()
        seed_default_user()
        init_video_router(db, get_whisper_model)
    except Exception as e:
        logger.error(f"Error during database seeding: {e}")

# ==========================================
# AUTH ENDPOINTS
# ==========================================
@app.post("/auth/register")
def register_user(req: RegisterRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    users_collection = db["users"]
    
    # Check if email already exists
    if users_collection.find_one({"email": req.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    hashed_pwd = hash_password(req.password)
    
    try:
        users_collection.insert_one({
            "_id": user_id,
            "name": req.name,
            "email": req.email,
            "password_hash": hashed_pwd,
            "streak_count": 0,
            "last_practice_date": None,
            "created_at": datetime.utcnow()
        })
        logger.info(f"User '{req.email}' successfully registered.")
        return {"status": "success", "message": "User registered successfully"}
    except Exception as e:
        logger.error(f"Failed to register user: {e}")
        raise HTTPException(status_code=500, detail="Failed to register user")

@app.post("/auth/login")
def login_user(req: LoginRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    users_collection = db["users"]
    user = users_collection.find_one({"email": req.email})
    
    if not user or not user.get("password_hash"):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Invalid email or password")
    
    user_id = user["_id"]
    token = generate_token(user_id)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "name": user["name"],
            "email": user["email"]
        }
    }

# ==========================================
# API ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "SpeechAI API",
        "timestamp": datetime.utcnow().isoformat(),
        "database_connected": db is not None
    }

# ==========================================
# LLM HELPERS (Groq for Text Generation + Gemini for Evaluation & Suggestions)
# ==========================================
async def generate_custom_text_with_groq(
    prompt: str,
    system_prompt: str = "You are a professional speech therapy assistant. Output raw practice text only."
) -> tuple[Optional[str], Optional[str]]:
    """
    Uses Groq API Key to generate customized speech practice texts.
    Falls back gracefully to Gemini or other providers if Groq is unavailable.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    # 1. Primary: Query Groq Cloud API
    if groq_key:
        groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "allam-2-7b"]
        for model in groq_models:
            try:
                logger.info(f"Generating customized text via Groq API (model={model})...")
                async with httpx.AsyncClient(timeout=15.0) as client:
                    res = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {groq_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.7
                        }
                    )
                    if res.status_code == 200:
                        data = res.json()
                        text = data["choices"][0]["message"]["content"].strip()
                        if text.startswith('"') and text.endswith('"'):
                            text = text[1:-1]
                        logger.info(f"Successfully generated customized text from Groq ({model}).")
                        return text, "Groq Cloud"
                    else:
                        logger.warning(f"Groq API ({model}) returned status {res.status_code}: {res.text[:120]}")
            except Exception as e:
                logger.warning(f"Error querying Groq API with {model}: {e}")

    # Fallback to Gemini if Groq unavailable
    if gemini_key:
        for model in ["gemini-2.5-flash", "gemini-1.5-flash"]:
            try:
                logger.info(f"Fallback text generation via Google Gemini ({model})...")
                async with httpx.AsyncClient(timeout=15.0) as client:
                    full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                    res = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        headers={
                            "x-goog-api-key": gemini_key,
                            "Content-Type": "application/json"
                        },
                        json={"contents": [{"parts": [{"text": full_text}]}]}
                    )
                    if res.status_code == 200:
                        candidates = res.json().get("candidates", [])
                        if candidates:
                            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                            if text:
                                if text.startswith('"') and text.endswith('"'):
                                    text = text[1:-1]
                                return text, "Google Gemini"
            except Exception as e:
                logger.warning(f"Error in Gemini fallback text generation: {e}")

    return None, None


async def evaluate_results_with_gemini(
    transcript: str,
    wpm: float,
    filler_count: int,
    filler_words_found: list[str],
    stammer_events: int,
    long_pauses: int,
    target_sentence: Optional[str] = None,
    wer: Optional[float] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Uses Gemini API Key to evaluate speech session results and provide small and concise suggestions.
    Falls back gracefully to Groq or rule-based suggestions if Gemini is unavailable.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    target_info = ""
    if target_sentence:
        target_info = f"\n- Target sentence was: \"{target_sentence}\""
        if wer is not None:
            target_info += f"\n- Word error rate: {round(wer * 100, 1)}%"

    fillers_str = ", ".join(filler_words_found) if filler_words_found else "none"

    eval_prompt = (
        "You are an expert speech-language pathologist and communication coach.\n"
        "Evaluate the speaker's performance from their session metrics:\n"
        f"- Spoken transcript: \"{transcript}\"\n"
        f"- Speed: {round(wpm, 1)} WPM (optimal conversational pace: 130-150 WPM)\n"
        f"- Filler words: {filler_count} ({fillers_str})\n"
        f"- Hesitations / stammers: {stammer_events} instances\n"
        f"- Gaps of silence: {long_pauses} pauses{target_info}\n\n"
        "Provide a small, concise evaluation with 2-3 short, highly actionable suggestions for improvement. "
        "Format as 1 short assessment sentence followed by 2-3 concise bullet points. "
        "Keep your response under 60-75 words total, direct and encouraging. "
        "Do not include any headers, greeting, intro, outro, or quotes."
    )

    # 1. Primary: Google Gemini API (gemini-2.5-flash / gemini-1.5-flash)
    if gemini_key:
        for model in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]:
            try:
                logger.info(f"Evaluating speech results using Google Gemini API ({model})...")
                async with httpx.AsyncClient(timeout=15.0) as client:
                    res = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        headers={
                            "x-goog-api-key": gemini_key,
                            "Content-Type": "application/json"
                        },
                        json={"contents": [{"parts": [{"text": eval_prompt}]}]}
                    )
                    if res.status_code == 200:
                        candidates = res.json().get("candidates", [])
                        if candidates:
                            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                            if text:
                                if text.startswith('"') and text.endswith('"'):
                                    text = text[1:-1]
                                logger.info(f"Successfully received Gemini evaluation results ({model}).")
                                return text, "Google Gemini"
                    else:
                        logger.warning(f"Gemini API ({model}) returned status {res.status_code}: {res.text[:120]}")
            except Exception as e:
                logger.warning(f"Error querying Gemini API ({model}): {e}")

    # 2. Fallback: Groq Cloud
    if groq_key:
        try:
            logger.info("Evaluating speech results with Groq fallback...")
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={
                        "model": "openai/gpt-oss-120b",
                        "messages": [
                            {"role": "system", "content": "You are a professional speech pathologist. Provide small, concise evaluation and suggestions."},
                            {"role": "user", "content": eval_prompt}
                        ],
                        "temperature": 0.5
                    }
                )
                if res.status_code == 200:
                    text = res.json()["choices"][0]["message"]["content"].strip()
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    return text, "Groq Cloud"
        except Exception as e:
            logger.warning(f"Error in Groq evaluation fallback: {e}")

    # 3. Rule-based small and concise fallback suggestions
    suggestions = []
    if wpm < 110 and wpm > 0:
        suggestions.append("• Aim for a slightly faster, more rhythmic pace (130-150 WPM).")
    elif wpm > 155:
        suggestions.append("• Slow your pace slightly to give articulation space to breathe.")
    else:
        suggestions.append("• Maintain your steady, natural speaking pace.")

    if filler_count > 2:
        suggestions.append(f"• Replace filler words like '{filler_words_found[0] if filler_words_found else 'um'}' with brief silent pauses.")
    if long_pauses > 2:
        suggestions.append("• Outline your ideas before speaking to reduce hesitation gaps.")
    elif stammer_events > 1:
        suggestions.append("• Elongate initial vowels to ease through articulatory blocks.")

    if len(suggestions) < 3:
        suggestions.append("• Practice diaphragmatic breathing before starting each sentence.")

    fallback_eval = (
        f"Your pace of {round(wpm, 1)} WPM demonstrates steady communication. "
        "Suggestions:\n" + "\n".join(suggestions[:3])
    )
    return fallback_eval, "Local Evaluation"


async def query_llm(
    prompt: str,
    system_prompt: str = "You are a professional speech therapy assistant.",
    prefer: str = "auto"
) -> tuple[Optional[str], Optional[str]]:
    """
    General query LLM helper supporting 'groq', 'gemini', or 'auto' provider preference.
    """
    if prefer == "groq":
        return await generate_custom_text_with_groq(prompt, system_prompt)
    elif prefer == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            for model in ["gemini-2.5-flash", "gemini-1.5-flash"]:
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        res = await client.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                            headers={"x-goog-api-key": gemini_key, "Content-Type": "application/json"},
                            json={"contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}]}
                        )
                        if res.status_code == 200:
                            candidates = res.json().get("candidates", [])
                            if candidates:
                                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                                return text, "Google Gemini"
                except Exception as e:
                    logger.warning(f"Error querying Gemini: {e}")
        return await generate_custom_text_with_groq(prompt, system_prompt)
    else:
        # Default auto order
        return await generate_custom_text_with_groq(prompt, system_prompt)


@app.get("/practice/generate")
async def generate_practice_text(
    topic: Optional[str] = "General Communication",
    length: Optional[str] = "paragraph",
    exercise_id: Optional[str] = "none",
    level: Optional[str] = "medium",
    difficulty: Optional[str] = None
):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    level_val = (difficulty or level or "medium").lower()
    if level_val == "easy":
        level_instruction = (
            "The English difficulty level must be EASY (beginner friendly): use simple vocabulary, short and straightforward sentences, "
            "and familiar everyday words suitable for basic language learners."
        )
    elif level_val == "difficult":
        level_instruction = (
            "The English difficulty level must be DIFFICULT (advanced/mastery): use sophisticated and challenging vocabulary, complex sentence structures, "
            "diverse linguistic phrasing, and rich rhetorical expressions suitable for advanced fluent speakers."
        )
    else:
        level_val = "medium"
        level_instruction = (
            "The English difficulty level must be MEDIUM (intermediate): use standard conversational and professional vocabulary "
            "with balanced sentence complexity and natural phrasing."
        )

    focus_guidelines = ""
    if exercise_id and exercise_id != "none":
        try:
            ex = db["exercises"].find_one({"_id": exercise_id})
            if ex:
                focus_guidelines = f" Focus on the requirements of the exercise '{ex['title']}': {ex['description']}."
        except Exception as e:
            logger.error(f"Error fetching exercise context: {e}")

    length_desc = "one single sentence of about 10-15 words"
    if length == "paragraph":
        length_desc = "one paragraph of about 40-60 words"
    elif length == "long_paragraph":
        length_desc = "one extended paragraph or two of about 100-150 words"

    prompt = (
        f"You are a helpful speech therapist assistant. Generate a customized speech practice text "
        f"about the topic '{topic}'. The text must be exactly {length_desc}. {level_instruction}{focus_guidelines} "
        f"Ensure the text is natural, engaging, and contains vocabulary relevant to the topic. "
        f"Do not include any intro, outro, titles, quotes, markdown formatting, or metadata. Output ONLY the raw text to be read aloud."
    )

    # Use Groq API Key to generate customized text
    generated_text, source_name = await generate_custom_text_with_groq(
        prompt=prompt,
        system_prompt="You are a professional speech therapist assistant. Output raw practice text only."
    )
    if generated_text:
        return {"text": generated_text, "source": source_name, "level": level_val}

    logger.info("Using local template generator fallback.")
    fallback_templates = {
        "easy": {
            "sentence": [
                "Good communication with our friends helps us share happy stories.",
                "Cooking healthy food every day gives our bodies energy and strength.",
                "New technology helps people talk to each other across the world.",
                "Speaking clearly and slowly helps everyone understand your ideas.",
                "Learning new words every day is fun and exciting."
            ],
            "paragraph": [
                "Speaking to other people can be easy and fun when you relax. Take a slow breath before you start speaking. Keep your words clear and say them at a calm pace. When you practice every day, speaking becomes much easier and you will feel more confident.",
                "Making food at home is a great way to learn new skills. You can wash fresh vegetables, cut them carefully, and put them in a warm pot. When the food is ready, eating dinner with your family makes everyone feel happy.",
                "Looking up at the night sky makes people think about the stars and planets. Scientists build rockets to visit space and learn new things. Discovering what is far away helps us understand our own home on Earth."
            ],
            "long_paragraph": [
                "When you want to speak clearly, the best step is to stay calm and take deep breaths. Many people speak too fast when they feel nervous, and this makes it hard for listeners to follow. By pausing quietly between sentences, you give yourself time to think of the next word. Reading stories out loud every morning helps you practice saying each word smoothly and building lasting confidence."
            ]
        },
        "medium": {
            "sentence": [
                "We should discuss the importance of communication in space exploration.",
                "Cooking requires patience, fresh ingredients, and a good understanding of recipes.",
                "Technology continues to evolve rapidly, transforming the way we connect with others.",
                "Professional success is built on active listening and concise speech articulation.",
                "Learning a new language is a beautiful journey that enriches the human mind."
            ],
            "paragraph": [
                "Speaking in front of an audience can be intimidating at first, but with steady pacing and deliberate breaths, anyone can deliver a powerful message. It is essential to focus on articulation and maintain a conversational speed of around one hundred and thirty words per minute, avoiding filler words.",
                "The culinary arts offer a wonderful blend of creativity and science. Whether you are baking bread or slow-cooking a savory soup, each step requires attention to detail. Sharing a warm meal with family and friends is one of the oldest and most universal ways of expressing care.",
                "Human exploration of outer space has inspired generations of scientists, writers, and dreamers. Sending satellites into orbit and landing rovers on Mars helps us answer fundamental questions about our solar system. The journey to the stars is a testament to human curiosity and innovation."
            ],
            "long_paragraph": [
                "To speak clearly and confidently, one must practice the art of breath control and deliberate pacing. A common mistake is rushing through sentences, which leads to slurred consonants and frequent stammering. By breaking your speech into logical chunks and pausing silently for a brief moment between key ideas, you give the listener time to absorb your thoughts. Daily drills focusing on challenging tongue-twisters and steady reading exercises will significantly boost your overall articulation, clarity, and voice resonance over time.",
                "Modern digital technology plays an indispensable role in shaping our daily routines, from remote work platforms to automated smart home devices. While these advancements offer incredible convenience and boost global productivity, they also challenge us to find a healthy balance between online interactions and face-to-face communication. As artificial intelligence and machine learning models continue to mature, the key focus remains on leveraging technology ethically to solve real-world problems and improve lives."
            ]
        },
        "difficult": {
            "sentence": [
                "Articulating nuanced geopolitical discourse necessitates unparalleled linguistic dexterity and rhetorical finesse.",
                "Gastronomic sophistication demands meticulous harmonization of delicate herbs, quintessential seasoning, and culinary precision.",
                "Technological paradigm shifts persistently revolutionize socio-economic infrastructures and multilateral paradigms.",
                "Cultivating acoustic resonance and impeccable phonetic enunciation facilitates persuasive eloquence in executive leadership.",
                "Cognitive linguistics illuminates how syntactic intricacies fundamentally sculpt human consciousness and cultural perception."
            ],
            "paragraph": [
                "Oratorical prowess transcends mere verbal competence, demanding scrupulous modulation of intonation, cadenced respiration, and authoritative presence. When navigating multifaceted controversies, seasoned rhetoricians deliberately harness strategic pauses, systematically dismantling cognitive dissonance while enunciating intricate syllables with consummate precision and unyielding composure.",
                "Gastronomic alchemy represents an exquisite confluence of empirical chemistry and aesthetic craftsmanship. Cultivating an epicurean palate requires deciphering subtleties of caramelization, enzymatic tenderization, and harmonic umami profiles, transforming rudimentary sustenance into transcendent sensory spectacles that commemorate culinary heritage.",
                "Astrophysical exploration constitutes humanity's ultimate endeavor to decipher celestial mechanics and cosmological mysteries. Scrutinizing planetary atmospheres and gravitational anomalies broadens scientific paradigms, challenging existential presumptions while underscoring our infinitesimal footprint within the primordial expanse of spacetime."
            ],
            "long_paragraph": [
                "Mastering sophisticated verbal articulation demands rigorous discipline in vocal acoustics, breath management, and diaphragmatic projection. Impetuous speakers frequently succumb to phonetic degradation, obscuring polysyllabic transitions and diluting rhetoric through repetitive fillers. By orchestrating strategic pauses and deliberate cadence, the speaker commands authority and imbues prose with gravitas, ensuring complex philosophical deliberations are received with clarity, elegance, and persuasive resonance."
            ]
        }
    }

    tier_dict = fallback_templates.get(level_val, fallback_templates["medium"])
    selected_list = tier_dict.get(length, tier_dict.get("sentence", []))
    
    matched = [t for t in selected_list if topic.lower() in t.lower()]
    import random
    text = random.choice(matched) if matched else random.choice(selected_list)
    
    if exercise_id == "articulation_drill":
        text += " She sells seashells by the seashore, and the seashells she sells are surely seashells."

    return {"text": text, "source": "local_fallback", "level": level_val}

@app.get("/exercises")
def get_exercises():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    try:
        exercises = list(db["exercises"].find({}))
        for ex in exercises:
            ex["id"] = str(ex["_id"])
        return exercises
    except Exception as e:
        logger.error(f"Error fetching exercises: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def compute_user_analytics(sessions: List[Dict[str, Any]], streak_count: int = 0) -> Dict[str, Any]:
    """
    Computes statistical speech analytics across historical practice & analysis sessions.
    - Chronological timeline tracking
    - Baseline (first 3 sessions) vs Recent (last 3 sessions) comparisons
    - % improvements for Fluency, Accuracy, Filler reduction, and Pacing
    - OLS linear regression trajectory slope
    - Compound Speech Mastery Index (0-100)
    - 5-dimension radar competency comparison
    - Weak words mastery tracker
    """
    if not sessions:
        return {
            "has_data": False,
            "total_sessions": 0,
            "streak_count": streak_count,
            "fluency": {"current": 0.0, "baseline": 0.0, "pct_change": 0.0, "slope": 0.0, "status": "No practice logs yet"},
            "accuracy": {"current": 0.0, "baseline": 0.0, "pct_change": 0.0},
            "fillers": {"current_fpm": 0.0, "baseline_fpm": 0.0, "pct_reduction": 0.0},
            "pacing": {"current_wpm": 0.0, "status": "No data"},
            "mastery_index": {"score": 0, "tier": "Foundational", "label": "Begin Practice"},
            "radar_data": [],
            "timeline": [],
            "weak_words": []
        }

    # Chronological sort (oldest to newest)
    chrono = sorted(sessions, key=lambda s: str(s.get("created_at", "")))
    n = len(chrono)
    k = min(3, n)

    # 1. Fluency scores
    scores = []
    for s in chrono:
        score = s.get("final_score")
        if score is None:
            score = s.get("pronunciation_score", 0.0)
        try:
            scores.append(float(score or 0.0))
        except (ValueError, TypeError):
            scores.append(0.0)

    base_fluency = round(sum(scores[:k]) / k, 1)
    recent_fluency = round(sum(scores[-k:]) / k, 1)
    fluency_pct = round(((recent_fluency - base_fluency) / base_fluency) * 100, 1) if base_fluency > 0 else 0.0

    # Ordinary Least Squares (OLS) slope
    slope = 0.0
    if n >= 2:
        x_bar = (n - 1) / 2.0
        y_bar = sum(scores) / n
        num = sum((i - x_bar) * (scores[i] - y_bar) for i in range(n))
        den = sum((i - x_bar) ** 2 for i in range(n))
        slope = round(num / den, 3) if den != 0 else 0.0

    if slope > 0.12:
        fluency_status = "Accelerating Progress"
    elif slope < -0.12:
        fluency_status = "Needs Targeted Drills"
    else:
        fluency_status = "Steady Cadence"

    # 2. Pronunciation Accuracy (% derived from WER or pronunciation_score)
    accuracies = []
    for s in chrono:
        wer = s.get("word_error_rate")
        if wer is not None:
            try:
                acc = max(0.0, min(100.0, (1.0 - float(wer)) * 100.0))
            except (ValueError, TypeError):
                acc = 75.0
        elif s.get("pronunciation_score") is not None:
            try:
                acc = min(100.0, float(s["pronunciation_score"]) * 10.0)
            except (ValueError, TypeError):
                acc = 75.0
        else:
            try:
                acc = min(100.0, float(s.get("final_score", 7.0)) * 10.0)
            except (ValueError, TypeError):
                acc = 70.0
        accuracies.append(round(acc, 1))

    base_acc = round(sum(accuracies[:k]) / k, 1)
    recent_acc = round(sum(accuracies[-k:]) / k, 1)
    acc_pct = round(((recent_acc - base_acc) / max(1.0, base_acc)) * 100, 1)

    # 3. Fillers (per minute rate)
    fpm_list = []
    for s in chrono:
        try:
            fc = float(s.get("filler_count", 0))
            dur_min = max(0.15, float(s.get("duration_sec", 60.0)) / 60.0)
            fpm_list.append(round(fc / dur_min, 1))
        except (ValueError, TypeError):
            fpm_list.append(0.0)

    base_fpm = round(sum(fpm_list[:k]) / k, 1)
    recent_fpm = round(sum(fpm_list[-k:]) / k, 1)
    filler_red_pct = round(((base_fpm - recent_fpm) / base_fpm) * 100, 1) if base_fpm > 0 else 0.0

    # 4. Pacing (WPM)
    wpm_list = []
    for s in chrono:
        try:
            wpm_list.append(float(s.get("wpm") or 0.0))
        except (ValueError, TypeError):
            wpm_list.append(0.0)

    valid_wpms = [w for w in wpm_list if w > 0]
    recent_wpm = round(sum(valid_wpms[-k:]) / len(valid_wpms[-k:]), 1) if valid_wpms else 130.0

    if 120 <= recent_wpm <= 150:
        pacing_status = "Optimal Target Cadence (120-150 WPM)"
    elif recent_wpm < 120:
        pacing_status = "Deliberate / Slow Pace (<120 WPM)"
    else:
        pacing_status = "Brisk / Rapid Pace (>150 WPM)"

    # 5. Compound Speech Mastery Index (CSMI, 0-100)
    c_fl = (recent_fluency / 10.0) * 30.0
    c_acc = (recent_acc / 100.0) * 25.0
    recent_sub = chrono[-1].get("sub_scores", {}) if isinstance(chrono[-1].get("sub_scores"), dict) else {}
    c_pause = min(20.0, (float(recent_sub.get("pause_score", 7.0)) + float(recent_sub.get("stammer_score", 7.0))) / 20.0 * 20.0)
    c_filler = min(15.0, (float(recent_sub.get("filler_score", 7.0)) / 10.0) * 15.0)
    c_streak = min(10.0, (streak_count / 7.0) * 10.0)
    mastery_score = int(min(100, max(15, round(c_fl + c_acc + c_pause + c_filler + c_streak))))

    if mastery_score >= 85:
        tier, label = "Master Speaker", "Articulate & Highly Resilient"
    elif mastery_score >= 70:
        tier, label = "Proficient", "Confident & Fluid Cadence"
    elif mastery_score >= 50:
        tier, label = "Developing", "Consistent Progress & Clarity"
    else:
        tier, label = "Foundational", "Building Rhythm & Breath Control"

    # 6. 5-Dimension Radar Competency (0-10 Scale)
    base_sub = chrono[0].get("sub_scores", {}) if isinstance(chrono[0].get("sub_scores"), dict) else {}
    radar_data = [
        {"dimension": "Articulation", "baseline": round(base_acc / 10.0, 1), "current": round(recent_acc / 10.0, 1), "fullMark": 10},
        {"dimension": "Pacing & Rhythm", "baseline": max(3.0, round(10.0 - abs((wpm_list[0] if wpm_list else 125) - 135) / 8.0, 1)), "current": max(3.0, round(10.0 - abs(recent_wpm - 135) / 8.0, 1)), "fullMark": 10},
        {"dimension": "Pause Control", "baseline": float(base_sub.get("pause_score", 6.0)), "current": float(recent_sub.get("pause_score", 8.0)), "fullMark": 10},
        {"dimension": "Phonetic Clarity", "baseline": float(base_sub.get("clarity_score", 6.5)), "current": float(recent_sub.get("clarity_score", 8.5)), "fullMark": 10},
        {"dimension": "Filler Resistance", "baseline": float(base_sub.get("filler_score", 5.5)), "current": float(recent_sub.get("filler_score", 8.0)), "fullMark": 10},
    ]

    # 7. Timeline points with EMA (Exponential Moving Average)
    ema = scores[0] if scores else 0.0
    alpha = 0.35
    timeline = []
    for idx, s in enumerate(chrono):
        sc = scores[idx]
        ema = alpha * sc + (1 - alpha) * ema
        date_raw = s.get("created_at", "")
        try:
            dt = datetime.fromisoformat(str(date_raw).replace("Z", "+00:00"))
            date_label = dt.strftime("%b %d, %H:%M")
        except Exception:
            date_label = f"Sess #{idx + 1}"

        timeline.append({
            "name": date_label,
            "session_id": str(s.get("session_id") or s.get("_id") or f"sess_{idx}"),
            "type": s.get("type", "analysis"),
            "fluency": sc,
            "ema_trend": round(ema, 2),
            "accuracy": accuracies[idx],
            "wpm": wpm_list[idx],
            "fillers": int(s.get("filler_count", 0)),
            "stammers": int(s.get("stammer_events", 0)),
            "pauses": int(s.get("long_pauses", 0)),
        })

    # 8. Weak words & Pronunciation Shelf
    word_freq = {}
    last_mismatches = set(m.get("expected", "").lower() for m in chrono[-1].get("mismatched_words", [])) if chrono[-1].get("mismatched_words") else set()
    for s in chrono:
        for m in s.get("mismatched_words", []):
            w = m.get("expected", "").strip().lower()
            if w and len(w) > 2:
                if w not in word_freq:
                    word_freq[w] = {"word": w, "count": 0, "spoken_variants": set()}
                word_freq[w]["count"] += 1
                if m.get("spoken"):
                    word_freq[w]["spoken_variants"].add(m.get("spoken"))

    weak_words = []
    for w, data in sorted(word_freq.items(), key=lambda x: x[1]["count"], reverse=True)[:8]:
        is_still_flagged = w in last_mismatches
        status = "needs_drill" if is_still_flagged else "mastered"
        weak_words.append({
            "word": w,
            "count": data["count"],
            "spoken_as": list(data["spoken_variants"])[:2],
            "status": status
        })

    return {
        "has_data": True,
        "total_sessions": n,
        "streak_count": streak_count,
        "fluency": {
            "current": recent_fluency,
            "baseline": base_fluency,
            "pct_change": fluency_pct,
            "slope": slope,
            "status": fluency_status
        },
        "accuracy": {
            "current": recent_acc,
            "baseline": base_acc,
            "pct_change": acc_pct
        },
        "fillers": {
            "current_fpm": recent_fpm,
            "baseline_fpm": base_fpm,
            "pct_reduction": filler_red_pct
        },
        "pacing": {
            "current_wpm": recent_wpm,
            "status": pacing_status
        },
        "mastery_index": {
            "score": mastery_score,
            "tier": tier,
            "label": label
        },
        "radar_data": radar_data,
        "timeline": timeline,
        "weak_words": weak_words
    }

@app.get("/reports/{user_id}")
def get_user_reports(user_id: str, authorization: Optional[str] = Header(None)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    
    # Overwrite user_id if token is valid
    auth_user_id = get_user_id_from_header(authorization)
    target_user_id = auth_user_id if auth_user_id != "user_anonymous" else user_id

    try:
        practice_cursor = db["practice_sessions"].find({"user_id": target_user_id}).sort("created_at", -1)
        practice_sessions = list(practice_cursor)
        for s in practice_sessions:
            s["_id"] = str(s["_id"])
            s["type"] = s.get("session_category", "practice")
            s["created_at"] = serialize_datetime(s["created_at"]) if isinstance(s["created_at"], datetime) else s["created_at"]

        analysis_cursor = db["analysis_sessions"].find({"user_id": target_user_id}).sort("created_at", -1)
        analysis_sessions = list(analysis_cursor)
        for s in analysis_sessions:
            s["_id"] = str(s["_id"])
            s["type"] = s.get("session_category", "analysis")
            s["created_at"] = serialize_datetime(s["created_at"]) if isinstance(s["created_at"], datetime) else s["created_at"]

        video_cursor = db["video_sessions"].find({"user_id": target_user_id, "status": "COMPLETED"}).sort("created_at", -1)
        video_sessions = list(video_cursor)
        for s in video_sessions:
            job_id_str = str(s.get("_id", s.get("job_id", "")))
            s["_id"] = job_id_str
            s["id"] = job_id_str
            s["session_id"] = str(s.get("session_id", job_id_str))
            s["type"] = "video"
            scores = s.get("scores", {})
            if isinstance(scores, dict):
                if scores.get("overall_10") is not None:
                    s["final_score"] = scores.get("overall_10")
                elif scores.get("overall") is not None:
                    s["final_score"] = scores.get("overall")
            speech = s.get("speech", {})
            if isinstance(speech, dict):
                s["transcript"] = speech.get("transcript", s.get("transcript", ""))
                s["wpm"] = speech.get("wpm", s.get("wpm", 0))
                s["filler_count"] = speech.get("filler_count", s.get("filler_count", 0))
                s["long_pauses"] = speech.get("long_pauses", s.get("long_pauses", 0))
                s["stammer_events"] = speech.get("repetition_count", s.get("stammer_events", 0))
                if "pause_events" in speech and not s.get("pause_details"):
                    s["pause_details"] = speech.get("pause_events")
            s["created_at"] = serialize_datetime(s["created_at"]) if isinstance(s["created_at"], datetime) else str(s.get("created_at", ""))
            if isinstance(s.get("completed_at"), datetime):
                s["completed_at"] = serialize_datetime(s["completed_at"])

        combined = practice_sessions + analysis_sessions + video_sessions
        combined.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        user_record = db["users"].find_one({"_id": target_user_id})
        streak = user_record.get("streak_count", 0) if user_record else 0

        analytics = compute_user_analytics(combined, streak)

        return {
            "streak_count": streak,
            "sessions": combined,
            "analytics": analytics
        }
    except Exception as e:
        logger.error(f"Error fetching user reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{user_id}/analytics")
def get_user_analytics_endpoint(user_id: str, authorization: Optional[str] = Header(None)):
    data = get_user_reports(user_id=user_id, authorization=authorization)
    return data.get("analytics", {})

def align_words(target_words: List[str], spoken_words: List[str]) -> List[Dict[str, Any]]:
    """
    Performs anchor-based sequence alignment between target_words and spoken_words.
    Uses SequenceMatcher to lock down matching anchor words, and accurately pairs
    substituted, mispronounced, or omitted tokens without cascading false mismatches.
    """
    clean_t = [re.sub(r'[^a-z0-9]', '', w.lower()) for w in target_words]
    clean_s = [re.sub(r'[^a-z0-9]', '', w.lower()) for w in spoken_words]
    
    matcher = difflib.SequenceMatcher(None, clean_t, clean_s)
    mismatched = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'delete':
            # Target words omitted by speaker
            for idx in range(i1, i2):
                mismatched.append({
                    "expected": target_words[idx],
                    "spoken": "",
                    "index": idx
                })
        elif tag == 'insert':
            # Extra words inserted by speaker (fillers, hesitations, repetitions) - do not flag target words
            continue
        elif tag == 'replace':
            t_slice = target_words[i1:i2]
            t_clean_slice = clean_t[i1:i2]
            s_slice = spoken_words[j1:j2]
            s_clean_slice = clean_s[j1:j2]
            
            t_to_s = {}
            used_s = set()
            
            # 1. High similarity match (morphological/phonetic mispronunciation)
            for t_idx, (t_orig, t_c) in enumerate(zip(t_slice, t_clean_slice)):
                best_s_idx = -1
                best_score = 0.0
                for s_idx, (s_orig, s_c) in enumerate(zip(s_slice, s_clean_slice)):
                    if s_idx in used_s:
                        continue
                    sim = difflib.SequenceMatcher(None, t_c, s_c).ratio()
                    if t_c and s_c and t_c[0] == s_c[0]:
                        sim += 0.12
                    if sim > best_score:
                        best_score = sim
                        best_s_idx = s_idx
                
                if best_s_idx != -1 and best_score >= 0.40:
                    t_to_s[t_idx] = s_slice[best_s_idx]
                    used_s.add(best_s_idx)
            
            # 2. Equal length blocks: 1-to-1 sequential mapping for remaining
            if len(t_slice) == len(s_slice):
                for t_idx in range(len(t_slice)):
                    if t_idx not in t_to_s:
                        for s_idx in range(len(s_slice)):
                            if s_idx not in used_s:
                                t_to_s[t_idx] = s_slice[s_idx]
                                used_s.add(s_idx)
                                break
            
            # 3. If there are single target words or unused spoken words, assign appropriately
            for t_idx in range(len(t_slice)):
                if t_idx not in t_to_s:
                    for s_idx in range(len(s_slice)):
                        if s_idx not in used_s:
                            t_to_s[t_idx] = s_slice[s_idx]
                            used_s.add(s_idx)
                            break
                    if t_idx not in t_to_s:
                        t_to_s[t_idx] = ""
            
            for t_local_idx, t_orig in enumerate(t_slice):
                abs_idx = i1 + t_local_idx
                mismatched.append({
                    "expected": t_orig,
                    "spoken": t_to_s.get(t_local_idx, ""),
                    "index": abs_idx
                })

    return mismatched

@app.post("/analyze/speech")
async def analyze_speech(
    audio: UploadFile = File(...),
    target_sentence: Optional[str] = Form(None),
    exercise_id: Optional[str] = Form(None),
    exercise_title: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    """
    Module 1: Speech Fluency Tracker Endpoint
    """
    user_id = get_user_id_from_header(authorization)
    logger.info(f"Received speech analysis request for user '{user_id}' with exercise_id: '{exercise_id}'")

    session_id = str(uuid.uuid4())
    temp_raw_path = os.path.join(TEMP_DIR, f"{session_id}_{audio.filename}")
    temp_wav_path = os.path.join(TEMP_DIR, f"{session_id}_converted.wav")

    try:
        with open(temp_raw_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        if not convert_to_wav(temp_raw_path, temp_wav_path):
            raise HTTPException(status_code=500, detail="Audio conversion failed")

        model = get_whisper_model()
        result = model.transcribe(
            temp_wav_path,
            initial_prompt=WHISPER_FILLER_PROMPT,
            condition_on_previous_text=False,
            word_timestamps=True,
            language="en"
        )
        transcript = result.get("text", "").strip()
        segments = result.get("segments", [])

        target_clean = ""
        wer_val = None
        mismatched_words = []
        
        if target_sentence:
            target_clean = clean_text(target_sentence)
            spoken_clean = clean_text(transcript)
            if target_clean:
                try:
                    wer_val = jiwer.wer(target_clean, spoken_clean)
                    mismatched_words = align_words(target_clean.split(), spoken_clean.split())
                except Exception as e:
                    logger.error(f"Error computing WER in fluency analysis: {e}")

        # 1. Feature Extraction: Duration
        duration_sec = 0.0
        if segments:
            duration_sec = segments[-1].get("end", 0.0) - segments[0].get("start", 0.0)
        if duration_sec <= 0.0:
            duration_sec = 1.0

        # 2. Feature Extraction: Word Count
        words_list = []
        for segment in segments:
            if "words" in segment:
                words_list.extend(segment["words"])
        
        total_word_count = len(words_list)
        if total_word_count == 0:
            total_word_count = len(transcript.split())

        # 3. Feature Extraction: Speech Rate (WPM)
        duration_min = duration_sec / 60.0
        wpm = round(total_word_count / duration_min, 1) if duration_min > 0 else 0.0

        # 4. Feature Extraction: Filler Words (Lexical + Acoustic Filled Pauses)
        filler_count, filler_words_found, filler_details = detect_fillers(
            transcript=transcript,
            wav_path=temp_wav_path,
            words_list=words_list,
            return_details=True
        )

        # 5. Feature Extraction: Stammering
        stammer_events, stammer_details = detect_stammering(
            transcript=transcript,
            words_list=words_list,
            return_details=True
        )

        # 6. Feature Extraction: Long Pauses
        long_pauses = 0
        pause_details = []
        if len(words_list) > 1:
            for i in range(len(words_list) - 1):
                curr_end = words_list[i].get("end", 0.0)
                next_start = words_list[i+1].get("start", 0.0)
                gap = next_start - curr_end
                if gap > PAUSE_THRESHOLD_SEC:
                    long_pauses += 1
                    pause_details.append({
                        "start": round(curr_end, 1),
                        "end": round(next_start, 1),
                        "duration": round(gap, 1)
                    })

        # 7. Clarity Score
        avg_logprob = 0.0
        if segments:
            avg_logprob = sum(seg.get("avg_logprob", 0.0) for seg in segments) / len(segments)
        clarity_score = max(0.0, min(10.0, round((avg_logprob + 1.0) * 10, 1)))

        # --- SCORING ENGINE ---
        filler_rate = filler_count / max(1, total_word_count)
        filler_score = max(0.0, min(10.0, round(10.0 - (filler_rate * COEFF_FILLER), 1)))

        stammer_rate = stammer_events / max(1, total_word_count)
        stammer_score = max(0.0, min(10.0, round(10.0 - (stammer_rate * COEFF_STAMMER), 1)))

        pause_rate = long_pauses / max(1, total_word_count)
        pause_score = max(0.0, min(10.0, round(10.0 - (pause_rate * COEFF_PAUSE), 1)))

        if IDEAL_WPM_MIN <= wpm <= IDEAL_WPM_MAX:
            rate_score = 10.0
        elif wpm < IDEAL_WPM_MIN:
            rate_score = max(0.0, round(10.0 - ((IDEAL_WPM_MIN - wpm) / 12), 1))
        else:
            rate_score = max(0.0, round(10.0 - ((wpm - IDEAL_WPM_MAX) / 15), 1))

        final_score = round(
            (WEIGHT_FILLER * filler_score) +
            (WEIGHT_STAMMER * stammer_score) +
            (WEIGHT_PAUSE * pause_score) +
            (WEIGHT_RATE * rate_score) +
            (WEIGHT_CLARITY * clarity_score),
            1
        )

        # --- DATA-DRIVEN FEEDBACK & RECOMMENDATIONS ENGINE ---
        feedback = []
        recommended_exercises = []

        context = {
            "filler_score": filler_score,
            "stammer_score": stammer_score,
            "pause_score": pause_score,
            "clarity_score": clarity_score,
            "wpm": wpm
        }

        db_exercises = []
        if db is not None:
            try:
                db_exercises = list(db["exercises"].find({}))
            except Exception as e:
                logger.error(f"Failed to fetch exercises for evaluation: {e}")

        for ex in db_exercises:
            condition = ex.get("trigger_condition", "")
            if condition:
                if evaluate_condition(condition, context):
                    fb_msg = ex.get("feedback_message", "")
                    if fb_msg:
                        feedback.append(fb_msg)
                    recommended_exercises.append(ex.get("_id"))

        if not feedback:
            feedback.append("Excellent fluency! Your pace, pause usage, and articulation are in the optimal range.")
            recommended_exercises.append("advanced_impromptu_speaking")

        # Gemini Evaluation: Small, concise evaluation and actionable suggestions
        ai_pathologist_feedback, source_name = await evaluate_results_with_gemini(
            transcript=transcript,
            wpm=wpm,
            filler_count=filler_count,
            filler_words_found=filler_words_found,
            stammer_events=stammer_events,
            long_pauses=long_pauses,
            target_sentence=target_sentence,
            wer=wer_val
        )

        streak_count = update_user_streak(user_id)

        response_data = {
            "session_id": session_id,
            "transcript": transcript,
            "duration_sec": float(duration_sec),
            "word_count": int(total_word_count),
            "wpm": float(wpm),
            "filler_count": int(filler_count),
            "filler_words_found": filler_words_found,
            "filler_details": filler_details,
            "stammer_events": int(stammer_events),
            "stammer_details": stammer_details,
            "long_pauses": int(long_pauses),
            "pause_details": pause_details,
            "sub_scores": {
                "filler_score": float(filler_score),
                "stammer_score": float(stammer_score),
                "pause_score": float(pause_score),
                "rate_score": float(rate_score),
                "clarity_score": float(clarity_score)
            },
            "final_score": float(final_score),
            "feedback": feedback,
            "recommended_exercises": recommended_exercises,
            "ai_pathologist_feedback": ai_pathologist_feedback,
            "target_text": target_sentence,
            "word_error_rate": float(wer_val) if wer_val is not None else None,
            "mismatched_words": mismatched_words,
            "exercise_id": exercise_id,
            "exercise_title": exercise_title,
            "streak_count": streak_count
        }

        if db is not None:
            try:
                db["analysis_sessions"].insert_one({
                    **response_data,
                    "user_id": user_id,
                    "session_category": "exercise" if exercise_id else "analysis",
                    "created_at": datetime.utcnow()
                })
                logger.info(f"Speech analysis session '{session_id}' persisted to DB.")
            except Exception as e:
                logger.error(f"Failed to save speech analysis session to MongoDB: {e}")

        return response_data

    except Exception as e:
        logger.error(f"Error processing speech analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in (temp_raw_path, temp_wav_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file '{path}': {e}")


def normalize_tts_text(text: str) -> str:
    """Insert spaces into compact strings before sending them to TTS.

    Examples:
      'Thequickbrownfox...' -> 'The quick brown fox...'
      'helloWorld' -> 'hello World'
    """
    if not text:
        return text

    normalized = text.strip().replace('\u2019', "'")
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'([a-z])([A-Z])', r'\1 \2', normalized)
    normalized = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', normalized)
    normalized = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', normalized)
    normalized = re.sub(r'([,.!?;:])(?=[A-Za-z])', r'\1 ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


async def collect_tts_audio(stream_iterable):
    """Collect the complete MP3 payload before returning it to the browser.

    edge-tts streams audio in chunks. Returning those chunks as a live stream can
    cause browsers to start decoding before the initial MP3 frame set is complete,
    which often shows up as the first 1-2 seconds being skipped or missing.
    """
    chunks = []
    async for msg in stream_iterable:
        if msg.get('type') == 'audio' and msg.get('data'):
            chunks.append(msg['data'])
    return b''.join(chunks)


@app.get('/tts/generate')
async def generate_tts(text: Optional[str] = None, voice: Optional[str] = 'en-US-AriaNeural'):
    """Synthesize text to speech using edge-tts and return a complete MP3 blob.

    Example: GET /tts/generate?text=Hello+world
    """
    if not text:
        raise HTTPException(status_code=400, detail='Missing text parameter')

    text = normalize_tts_text(text)

    try:
        import edge_tts
    except Exception as e:
        logger.error(f'edge-tts is not available: {e}')
        raise HTTPException(status_code=500, detail='TTS engine not available on server')

    try:
        communicate = edge_tts.Communicate(text, voice=voice)
        audio_bytes = await collect_tts_audio(communicate.stream())
        if not audio_bytes:
            raise HTTPException(status_code=500, detail='No audio data was generated')
        return Response(content=audio_bytes, media_type='audio/mpeg')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error during TTS generation: {e}')
        raise HTTPException(status_code=500, detail='Failed to generate TTS audio')
