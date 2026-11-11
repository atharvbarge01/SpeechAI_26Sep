import os
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header, BackgroundTasks, status
from fastapi.responses import JSONResponse

from video_module.config import TEMP_DIR, MAX_UPLOAD_BYTES
from video_module.schemas import VideoJobResponse, VideoSessionDetail
from video_module.jobs import process_video_job, get_job_state, set_job_state

logger = logging.getLogger("Module2_Router")

router = APIRouter(prefix="/video", tags=["Module 2: Video Analysis"])

# Reference to the main database and Whisper model (injected during setup)
_db = None
_whisper_model_fn = None

def init_router(db_instance, whisper_model_getter):
    global _db, _whisper_model_fn
    _db = db_instance
    _whisper_model_fn = whisper_model_getter
    logger.info("Module 2 router initialized with shared DB and Whisper loader.")

def get_user_id(authorization: Optional[str]) -> str:
    if not authorization:
        return "user_anonymous"
    try:
        if authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            parts = token.split(":")
            if len(parts) == 3:
                return parts[0]
    except Exception:
        pass
    return "user_anonymous"

@router.get("/health")
def video_health():
    return {
        "status": "healthy",
        "service": "SpeechAI Module 2 (Video + Audio Communication Analysis)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database_available": _db is not None
    }

@router.post("/analyze", response_model=VideoJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_video_analysis(
    video: UploadFile = File(...),
    task_type: str = Form("free_talk"),
    prompt_text: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    """
    Submits a video recording for multimodal analysis.
    Returns 202 Accepted with a job_id for status polling.
    """
    user_id = get_user_id(authorization)
    job_id = str(uuid.uuid4())
    os.makedirs(TEMP_DIR, exist_ok=True)

    ext = os.path.splitext(video.filename)[1] if video.filename else ".webm"
    if not ext:
        ext = ".webm"
    temp_video_path = os.path.join(TEMP_DIR, f"{job_id}{ext}")

    try:
        bytes_written = 0
        with open(temp_video_path, "wb") as f:
            while chunk := await video.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Video file exceeds maximum 100 MB upload limit."
                    )
                f.write(chunk)
    except HTTPException:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise
    except Exception as e:
        logger.error(f"Failed to write uploaded video: {e}")
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise HTTPException(status_code=500, detail="Failed to receive video stream on server.")

    created_at = datetime.now(timezone.utc).isoformat()
    initial_job_data = {
        "job_id": job_id,
        "user_id": user_id,
        "status": "QUEUED",
        "stage": "queued",
        "created_at": created_at,
        "task_type": task_type,
        "prompt_text": prompt_text
    }
    set_job_state(job_id, initial_job_data)

    collection = _db["video_sessions"] if _db is not None else None
    if collection is not None:
        try:
            collection.insert_one({"_id": job_id, **initial_job_data})
        except Exception as e:
            logger.error(f"Failed to record queued video job in DB: {e}")

    # Launch async background worker
    whisper_model = _whisper_model_fn() if _whisper_model_fn else None
    asyncio.create_task(process_video_job(
        job_id=job_id,
        video_path=temp_video_path,
        user_id=user_id,
        task_type=task_type,
        prompt_text=prompt_text,
        db_collection=collection,
        whisper_model=whisper_model
    ))

    return {
        "job_id": job_id,
        "status": "QUEUED",
        "stage": "queued",
        "created_at": created_at
    }

@router.get("/session/{job_id}")
def get_session_status(job_id: str):
    """
    Polls the status or retrieves the full completed report of a video analysis session.
    """
    # 1. Check in-memory cache
    cached = get_job_state(job_id)
    if cached:
        return cached

    # 2. Check MongoDB
    if _db is not None:
        try:
            doc = _db["video_sessions"].find_one({"_id": job_id})
            if doc:
                doc["id"] = str(doc.get("_id", job_id))
                return doc
        except Exception as e:
            logger.error(f"DB lookup error for session {job_id}: {e}")

    raise HTTPException(status_code=404, detail="Video session not found.")

@router.get("/reports/{user_id}")
def get_user_video_reports(user_id: str, authorization: Optional[str] = Header(None)):
    """
    Returns past completed video analysis sessions for the specified user.
    """
    auth_user = get_user_id(authorization)
    target_user = auth_user if auth_user != "user_anonymous" else user_id

    if _db is None:
        return []

    try:
        cursor = _db["video_sessions"].find(
            {"user_id": target_user, "status": "COMPLETED"}
        ).sort("created_at", -1)
        
        sessions = []
        for doc in cursor:
            doc["id"] = str(doc.get("_id", ""))
            doc.pop("_id", None)
            sessions.append(doc)
        return sessions
    except Exception as e:
        logger.error(f"Failed to fetch video reports for user {user_id}: {e}")
        return []

@router.delete("/session/{job_id}")
def delete_video_session(job_id: str, authorization: Optional[str] = Header(None)):
    """
    Allows users to delete a saved session record.
    """
    if _db is None:
        raise HTTPException(status_code=500, detail="Database not connected.")

    user_id = get_user_id(authorization)
    res = _db["video_sessions"].delete_one({"_id": job_id, "user_id": user_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found or unauthorized.")
    return {"status": "deleted", "job_id": job_id}
