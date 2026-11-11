"""
Integration and unit test for the In-House Trained Speech Scoring Model.
Verifies checkpoint detection, loading, inference, and response format.
"""

import os
import sys
import asyncio

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ml_model.service import (
    find_checkpoint_and_scaler,
    is_local_model_available,
    get_scorer_instance,
    evaluate_speech_with_inhouse_model
)

def test_checkpoint_availability():
    ckpt, scaler = find_checkpoint_and_scaler()
    print(f"Detected checkpoint: {ckpt}")
    print(f"Detected scaler: {scaler}")
    assert ckpt is not None and os.path.exists(ckpt), "Checkpoint best_model.pt must exist"
    assert scaler is not None and os.path.exists(scaler), "scaler.json must exist"
    assert is_local_model_available() is True, "is_local_model_available() must be True"
    print("Checkpoint and scaler existence verified.")

def test_scorer_loading():
    scorer = get_scorer_instance()
    assert scorer is not None, "get_scorer_instance() should successfully instantiate model"
    print("Scorer loaded successfully on device:", scorer.device)

def _get_sample_audio_path():
    candidate_paths = [
        os.path.join(backend_dir, "..", "..", "dataset", "audio", "sample_001.wav"),
        os.path.join(backend_dir, "..", "dataset", "audio", "sample_001.wav"),
        os.path.join(backend_dir, "dataset", "audio", "sample_001.wav"),
    ]
    for p in candidate_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            return abs_p
    return None

def test_scorer_inference():
    sample_audio = _get_sample_audio_path()
    if sample_audio:
        scorer = get_scorer_instance()
        pred = scorer.predict(sample_audio)
        print(f"Inference on {sample_audio}:")
        for k, v in pred["parameters"].items():
            print(f"  - {k}: {v:.2f}")
        print(f"Composite score: {pred['composite_fluency_score']}")
        print(f"Feedback items: {len(pred['feedback'])}")
        
        # Verify required keys and ranges
        assert "parameters" in pred
        assert "composite_fluency_score" in pred
        for p in ["wpm", "filler_count", "stammer_count", "pause_count", "clarity", "wer"]:
            assert p in pred["parameters"], f"Missing metric {p}"
            assert pred["parameters"][p] >= 0, f"Metric {p} should be non-negative"
        print("Inference test passed successfully!")
    else:
        print("Sample audio not found, skipping direct audio inference test.")

def test_async_evaluate_service():
    sample_audio = _get_sample_audio_path()
    if sample_audio:
        fb, src, params = asyncio.run(evaluate_speech_with_inhouse_model(
            audio_path=sample_audio,
            transcript="The quick brown fox jumps over the lazy dog.",
            target_sentence="The quick brown fox jumps over the lazy dog.",
            wer=0.05
        ))
        assert src == "In-House Speech Scorer"
        assert fb is not None and len(fb) > 0
        assert params is not None and "wpm" in params
        print("Async service evaluation test passed successfully!")

if __name__ == "__main__":
    print("Testing In-House Speech Scorer ML pipeline...")
    test_checkpoint_availability()
    test_scorer_loading()
    test_scorer_inference()
    test_async_evaluate_service()
    print("ALL IN-HOUSE ML MODEL TESTS PASSED SUCCESSFULLY!")

