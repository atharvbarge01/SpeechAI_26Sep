import os
import shutil
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import cv2
import httpx

from video_module.media_utils import extract_audio, probe_video
from video_module.frame_sampler import sample_frames
from video_module.speech_features import extract_speech_features
from video_module.vision_features import analyze_frame, aggregate_visual_session
from video_module.windows import generate_fusion_timeline
from video_module.scoring import compute_speech_scores, compute_visual_scores, compute_overall_scores
from video_module.feedback import generate_multimodal_feedback

logger = logging.getLogger("Module2_Worker")

# In-memory job state cache for real-time polling updates
job_cache: Dict[str, Dict[str, Any]] = {}

def get_job_state(job_id: str) -> Optional[Dict[str, Any]]:
    return job_cache.get(job_id)

def set_job_state(job_id: str, updates: Dict[str, Any]):
    if job_id not in job_cache:
        job_cache[job_id] = {}
    job_cache[job_id].update(updates)

async def process_video_job(
    job_id: str,
    video_path: str,
    user_id: str,
    task_type: str,
    prompt_text: Optional[str],
    db_collection,
    whisper_model
):
    """
    Executes the full multimodal pipeline asynchronously:
    1. Extract Audio -> 2. Whisper Speech Features -> 3. FFmpeg Frame Sample & Vision -> 4. Fusion & Scoring -> 5. Store & Clean
    """
    start_time = time.time()
    timings: Dict[str, float] = {}
    temp_frames_dir = None
    temp_wav_path = None

    try:
        set_job_state(job_id, {"status": "PROCESSING", "stage": "extracting_audio"})

        # --- Stage 1: Audio Extraction & Probing ---
        t0 = time.time()
        meta = probe_video(video_path)
        duration_sec = meta.get("duration_sec", 0.0)

        temp_wav_path = f"{video_path}.wav"
        has_audio = extract_audio(video_path, temp_wav_path)
        timings["audio_extraction_sec"] = round(time.time() - t0, 2)

        # --- Stage 2: Speech Transcription (Whisper) ---
        t1 = time.time()
        set_job_state(job_id, {"stage": "transcribing"})

        speech_features = {
            "transcript": "",
            "word_count": 0,
            "wpm": 0.0,
            "filler_count": 0,
            "filler_types": [],
            "repetition_count": 0,
            "long_pauses": 0,
            "total_pause_time": 0.0,
            "pause_ratio": 0.0,
            "pause_events": [],
            "clarity_raw": -0.5,
            "words_list": []
        }

        if has_audio and whisper_model:
            try:
                initial_prompt = "Um, uh, umm, uhh, er, ah, hmm, like, you know, actually, basically, so, well, i mean."
                whisper_res = whisper_model.transcribe(
                    temp_wav_path,
                    initial_prompt=initial_prompt,
                    condition_on_previous_text=False,
                    word_timestamps=True,
                    language="en"
                )
                speech_features = extract_speech_features(whisper_res, duration_sec, wav_path=temp_wav_path)
                if duration_sec <= 0.0 and speech_features.get("words_list"):
                    duration_sec = speech_features["words_list"][-1].get("end", 1.0)
            except Exception as e:
                logger.error(f"Whisper processing error in job {job_id}: {e}")

        timings["transcription_sec"] = round(time.time() - t1, 2)

        # --- Stage 3: Visual Analysis (5 FPS Frames & MediaPipe) ---
        t2 = time.time()
        set_job_state(job_id, {"stage": "video_analysis"})

        temp_frames_dir = f"{video_path}_frames"
        sampled_frames = sample_frames(video_path, temp_frames_dir, fps=5.0)

        frame_results = []
        for ts, fpath in sampled_frames:
            img_bgr = cv2.imread(fpath)
            if img_bgr is not None:
                res = analyze_frame(img_bgr, ts)
                frame_results.append(res)

        visual_features = aggregate_visual_session(frame_results)
        timings["video_analysis_sec"] = round(time.time() - t2, 2)

        # If duration is still 0, derive from sampled frames
        if duration_sec <= 0.0 and sampled_frames:
            duration_sec = sampled_frames[-1][0]
        duration_sec = max(1.0, round(duration_sec, 1))

        # --- Stage 4: 5-Second Window Fusion & Scoring Engine ---
        t3 = time.time()
        set_job_state(job_id, {"stage": "scoring"})

        timeline = generate_fusion_timeline(
            duration_sec=duration_sec,
            words_list=speech_features.get("words_list", []),
            pause_events=speech_features.get("pause_events", []),
            frame_metrics=frame_results,
            frame_emotions=[]
        )

        speech_scores = compute_speech_scores(speech_features, task_type=task_type)
        visual_scores = compute_visual_scores(visual_features)

        f_score = speech_scores.get("F") if has_audio and speech_features["word_count"] > 0 else None
        p_score = speech_scores.get("P") if has_audio and speech_features["word_count"] > 0 else None
        n_score = visual_scores.get("N")
        e_score = visual_scores.get("E")

        overall_scores = compute_overall_scores(
            f_score=f_score,
            p_score=p_score,
            n_score=n_score,
            e_score=e_score
        )

        feedback_items = generate_multimodal_feedback(speech_features, visual_features, overall_scores)

        # Gemini Evaluation: Small, concise multimodal suggestions
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                g_prompt = (
                    "You are an expert speech pathologist and body language coach. Evaluate this multimodal presentation session:\n"
                    f"- Spoken Pace: {speech_features.get('wpm', 0.0)} WPM, Filler count: {speech_features.get('filler_count', 0)}\n"
                    f"- Eye Contact: {int(visual_features.get('camera_facing_ratio', 0.0) * 100)}% to camera\n"
                    f"- Head Stability: {round(visual_features.get('head_motion', 0.0), 3)} motion index\n"
                    f"- Postural Composure: {int((visual_features.get('posture_ok_ratio') or 0.8) * 100)}% level\n"
                    f"- Overall Score: {overall_scores.get('overall_100', 80)}/100\n\n"
                    "Provide exactly 2 small, concise, actionable suggestions (under 25 words each). "
                    "Return each suggestion starting with a dash (-)."
                )
                async with httpx.AsyncClient(timeout=10.0) as client:
                    g_res = await client.post(
                        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                        headers={"x-goog-api-key": gemini_key, "Content-Type": "application/json"},
                        json={"contents": [{"parts": [{"text": g_prompt}]}]}
                    )
                    if g_res.status_code == 200:
                        parts = g_res.json().get("candidates", [])[0].get("content", {}).get("parts", [])
                        if parts:
                            lines = [l.strip().lstrip("-*• ") for l in parts[0].get("text", "").splitlines() if l.strip()]
                            for line in lines[:2]:
                                if line:
                                    feedback_items.append({
                                        "category": "speech" if any(w in line.lower() for w in ["pace", "pause", "filler", "word", "breath", "speed", "vocal"]) else "non_verbal",
                                        "severity": "tip",
                                        "message": f"Gemini Coach: {line}"
                                    })
            except Exception as e:
                logger.warning(f"Error querying Gemini for multimodal evaluation: {e}")

        timings["scoring_sec"] = round(time.time() - t3, 2)

        total_proc_time = round(time.time() - start_time, 2)
        rtf = round(total_proc_time / duration_sec, 2) if duration_sec > 0 else 1.0

        # Build clean session payload
        result_payload = {
            "id": job_id,
            "user_id": user_id,
            "status": "COMPLETED",
            "stage": "done",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "task_type": task_type,
            "prompt_text": prompt_text,
            "duration_sec": duration_sec,
            "speech": speech_features,
            "visual": visual_features,
            "timeline": timeline,
            "scores": overall_scores,
            "feedback": feedback_items,
            "quality_warnings": visual_features.get("quality_warnings", []),
            "timings": timings,
            "rtf": rtf
        }

        # Persist in MongoDB
        if db_collection is not None:
            try:
                db_collection.update_one(
                    {"_id": job_id},
                    {"$set": {**result_payload, "_id": job_id}},
                    upsert=True
                )
                logger.info(f"Session {job_id} successfully persisted to MongoDB.")
            except Exception as e:
                logger.error(f"Failed to persist session to MongoDB: {e}")

        set_job_state(job_id, result_payload)

    except Exception as e:
        logger.error(f"Error executing video job {job_id}: {e}", exc_info=True)
        set_job_state(job_id, {"status": "FAILED", "stage": "failed", "error": str(e)})
        if db_collection is not None:
            try:
                db_collection.update_one(
                    {"_id": job_id},
                    {"$set": {"status": "FAILED", "error": str(e), "completed_at": datetime.now(timezone.utc).isoformat()}}
                )
            except Exception:
                pass

    finally:
        # Privacy by Default: ALWAYS remove raw video and temporary frames/audio
        for path in (video_path, temp_wav_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Cleanup error for file {path}: {e}")

        if temp_frames_dir and os.path.exists(temp_frames_dir):
            try:
                shutil.rmtree(temp_frames_dir)
            except Exception as e:
                logger.warning(f"Cleanup error for dir {temp_frames_dir}: {e}")
