"""
Backend Service Adapter for SpeechAI FastAPI.
Provides a clean, drop-in replacement for the Gemini API call:
`evaluate_speech_with_model(audio_path, transcript, target_sentence, wer)`
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger("speech_scorer_service")

# Detect paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoints", "best_model.pt")
DEFAULT_SCALER = os.path.join(CURRENT_DIR, "checkpoints", "scaler.json")

# Alternative root-level checkpoint location
ROOT_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
ROOT_SCALER = os.path.join(PROJECT_ROOT, "checkpoints", "scaler.json")

_global_scorer_instance = None

def get_scorer_instance():
    global _global_scorer_instance
    if _global_scorer_instance is not None:
        return _global_scorer_instance
        
    ckpt = DEFAULT_CHECKPOINT if os.path.exists(DEFAULT_CHECKPOINT) else ROOT_CHECKPOINT
    scaler = DEFAULT_SCALER if os.path.exists(DEFAULT_SCALER) else ROOT_SCALER
    
    if os.path.exists(ckpt) and os.path.exists(scaler):
        try:
            from speech_scoring_model.inference import SpeechScorerInference
            logger.info(f"Loading In-house Trained Speech Scorer from {ckpt}...")
            _global_scorer_instance = SpeechScorerInference(
                checkpoint_path=ckpt,
                scaler_path=scaler,
                device="cpu"  # default to cpu on backend to preserve server GPU memory
            )
            logger.info("Successfully loaded in-house Speech Scorer model.")
            return _global_scorer_instance
        except Exception as e:
            logger.warning(f"Could not initialize local speech scorer: {e}")
            return None
    return None


def is_local_model_available() -> bool:
    """Returns True if trained model checkpoint exists."""
    return (os.path.exists(DEFAULT_CHECKPOINT) or os.path.exists(ROOT_CHECKPOINT))


async def evaluate_speech_with_model(
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
        
        # Override wer if ground-truth target_sentence was compared
        if wer is not None:
            params["wer"] = round(float(wer), 3)
            
        feedback_lines = results["feedback"]
        feedback_str = " ".join(feedback_lines)
        
        return feedback_str, "In-house Trained Speech Scorer", params
    except Exception as e:
        logger.error(f"In-house speech model inference error: {e}")
        return None, "model_error", None

