"""
Backend Service Adapter for SpeechAI FastAPI.
Provides in-house model evaluation with graceful fallback.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("speech_scorer_service")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

CANDIDATE_CHECKPOINTS = [
    os.path.join(CURRENT_DIR, "checkpoints", "best_model.pt"),
    os.path.join(CURRENT_DIR, "..", "checkpoints", "best_model.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
]

CANDIDATE_SCALERS = [
    os.path.join(CURRENT_DIR, "checkpoints", "scaler.json"),
    os.path.join(CURRENT_DIR, "..", "checkpoints", "scaler.json"),
    os.path.join(PROJECT_ROOT, "checkpoints", "scaler.json")
]

_global_scorer_instance = None

def find_checkpoint_and_scaler() -> Tuple[Optional[str], Optional[str]]:
    for ckpt in CANDIDATE_CHECKPOINTS:
        if os.path.exists(ckpt):
            for sc in CANDIDATE_SCALERS:
                if os.path.exists(sc):
                    return ckpt, sc
    return None, None

def is_local_model_available() -> bool:
    ckpt, sc = find_checkpoint_and_scaler()
    return ckpt is not None and sc is not None

def get_scorer_instance():
    global _global_scorer_instance
    if _global_scorer_instance is not None:
        return _global_scorer_instance
        
    ckpt, sc = find_checkpoint_and_scaler()
    if ckpt and sc:
        try:
            from ml_model.inference import SpeechScorerInference
            logger.info(f"Loading In-house Trained Speech Scorer from {ckpt}...")
            _global_scorer_instance = SpeechScorerInference(
                checkpoint_path=ckpt,
                scaler_path=sc,
                device="cpu"
            )
            logger.info("Successfully loaded in-house Speech Scorer model.")
            return _global_scorer_instance
        except Exception as e:
            logger.warning(f"Could not initialize local speech scorer: {e}")
            return None
    return None


async def evaluate_speech_with_inhouse_model(
    audio_path: str,
    transcript: str,
    target_sentence: Optional[str] = None,
    wer: Optional[float] = None
) -> Tuple[Optional[str], str, Optional[Dict[str, float]]]:
    """
    Evaluates speech recording using the in-house trained 6-parameter model.
    Returns:
        (feedback_text, source_name, predicted_parameters)
    """
    scorer = get_scorer_instance()
    if scorer is None:
        return None, "model_not_found", None
        
    try:
        results = scorer.predict(audio_path)
        params = results["parameters"]
        
        if wer is not None:
            params["wer"] = round(float(wer), 3)
            
        feedback_lines = results["feedback"]
        feedback_str = (
            f"Assessment: Composite fluency score is {results['composite_fluency_score']}/10. "
            "Actionable Suggestions:\n" + "\n".join(f"- {tip}" for tip in feedback_lines[:3])
        )
        
        return feedback_str, "In-House Speech Scorer", params
    except Exception as e:
        logger.error(f"In-house speech model inference error: {e}")
        return None, "model_error", None

