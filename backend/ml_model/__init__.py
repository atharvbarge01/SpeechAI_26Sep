"""
In-House Speech Scoring ML Model package for SpeechAI Backend.
"""
from ml_model.config import ScorerConfig
from ml_model.service import evaluate_speech_with_inhouse_model, is_local_model_available

# Backward-compatible alias
evaluate_speech_with_model = evaluate_speech_with_inhouse_model


