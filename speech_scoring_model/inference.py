"""
Inference script for the 6-parameter Speech Scoring Model.
Takes an input audio file (.wav, .mp3, etc.) and predicts:
1. Speech Rate (WPM)
2. Filler Word Count
3. Stammer / Repetition Count
4. Long Pause Count
5. Clarity Score (0-10)
6. Word Error Rate (WER: 0-1)
Together with composite fluency score and actionable clinical feedback.
"""

import os
import sys
import argparse
import numpy as np
import librosa
import torch
from typing import Dict, Any, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from speech_scoring_model.config import ScorerConfig
from speech_scoring_model.model import Wav2Vec2SpeechScorer
from speech_scoring_model.dataset import ParameterScaler

class SpeechScorerInference:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        scaler_path: str = "checkpoints/scaler.json",
        device: Optional[str] = None
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        # Load scaler
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
        self.scaler = ParameterScaler.load(scaler_path)
        
        # Load model checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint.get("config", ScorerConfig())
        self.model = Wav2Vec2SpeechScorer(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Loads and resamples audio to 16kHz mono."""
        waveform, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        max_samples = int(self.config.max_audio_length_seconds * self.config.sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        if np.max(np.abs(waveform)) > 1e-4:
            waveform = waveform / np.max(np.abs(waveform))
        return torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)

    def generate_feedback(self, metrics: Dict[str, float]) -> List[str]:
        """
        Generates clinical suggestions directly from predicted parameters,
        replacing the external Gemini API call.
        """
        suggestions = []
        wpm = metrics["wpm"]
        fillers = metrics["filler_count"]
        stammers = metrics["stammer_count"]
        pauses = metrics["pause_count"]
        clarity = metrics["clarity"]
        wer = metrics["wer"]

        # 1. Pace suggestion
        if wpm < 115:
            suggestions.append(f"Your pace is somewhat deliberate ({round(wpm, 1)} WPM). Try phrasing thoughts in continuous breath groups.")
        elif wpm > 155:
            suggestions.append(f"Speaking pace is rapid ({round(wpm, 1)} WPM). Intersperse intentional micro-pauses at punctuation marks.")
        else:
            suggestions.append(f"Excellent conversational tempo ({round(wpm, 1)} WPM) within the target therapeutic window.")

        # 2. Filler words suggestion
        if fillers > 3:
            suggestions.append(f"Detected {int(round(fillers))} filler vocalizations. Replace vocal fillers with silent pauses when formulating ideas.")

        # 3. Hesitation & Stammer suggestion
        if stammers > 2:
            suggestions.append(f"Noticed {int(round(stammers))} syllable repetitions. Practice light articulatory contact and gentle vocal onsets.")

        # 4. Articulation & Clarity suggestion
        if clarity < 6.5:
            suggestions.append(f"Clarity score is {round(clarity, 1)}/10. Emphasize final consonants and vowel elongation.")
        elif wer > 0.20:
            suggestions.append(f"Word error rate indicates phonetic misalignments ({round(wer*100, 1)}%). Try syllable tapping drills.")

        if not suggestions:
            suggestions.append("Outstanding vocal control and smooth articulatory fluency across all parameters.")

        return suggestions

    def predict(self, audio_path: str) -> Dict[str, Any]:
        """Runs end-to-end inference on an audio file."""
        input_tensor = self.preprocess_audio(audio_path).to(self.device)
        attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor, attention_mask=attention_mask)
            norm_predictions = outputs["predictions"].cpu().numpy()

        # Denormalize to physical values
        raw_preds = self.scaler.inverse_transform(norm_predictions)[0]
        
        metrics = {}
        for i, name in enumerate(self.config.parameter_names):
            metrics[name] = float(round(raw_preds[i], 2))

        # Compute composite fluency score (0.0 - 10.0 scale)
        # 1. Pace sub-score
        wpm = metrics["wpm"]
        if 120.0 <= wpm <= 150.0:
            s_rate = 10.0
        elif wpm < 120.0:
            s_rate = max(0.0, 10.0 - ((120.0 - wpm) / 10.0))
        else:
            s_rate = max(0.0, 10.0 - ((wpm - 150.0) / 12.0))

        # 2. Filler sub-score
        s_filler = max(0.0, min(10.0, 10.0 - (metrics["filler_count"] * 1.5)))

        # 3. Stammer sub-score
        s_stammer = max(0.0, min(10.0, 10.0 - (metrics["stammer_count"] * 2.0)))

        # 4. Pause sub-score
        s_pause = max(0.0, min(10.0, 10.0 - (metrics["pause_count"] * 1.2)))

        # 5. Clarity sub-score
        s_clarity = metrics["clarity"]

        # Composite Fluency (0-10)
        composite_score = round(
            0.20 * s_rate +
            0.20 * s_filler +
            0.25 * s_stammer +
            0.20 * s_pause +
            0.15 * s_clarity,
            1
        )

        feedback = self.generate_feedback(metrics)

        return {
            "parameters": metrics,
            "composite_fluency_score": composite_score,
            "feedback": feedback,
            "model_source": "In-house Wav2Vec2 Speech Scorer (Local)"
        }


def score_audio(
    audio_path: str,
    checkpoint_path: str = "checkpoints/best_model.pt",
    scaler_path: str = "checkpoints/scaler.json",
    device: Optional[str] = None
) -> Dict[str, Any]:
    scorer = SpeechScorerInference(checkpoint_path=checkpoint_path, scaler_path=scaler_path, device=device)
    return scorer.predict(audio_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio using trained Speech Scorer")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav/.mp3)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--scaler", type=str, default="checkpoints/scaler.json", help="Path to scaler json")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    results = score_audio(args.audio, args.checkpoint, args.scaler, args.device)
    
    print("\n" + "="*50)
    print(" INFERENCE RESULTS (In-house Local Model)")
    print("="*50)
    for k, v in results["parameters"].items():
        print(f" - {k.upper():<15}: {v}")
    print(f" - COMPOSITE SCORE: {results['composite_fluency_score']}/10.0")
    print("\nActionable Pathologist Suggestions:")
    for tip in results["feedback"]:
        print(f" * {tip}")
    print("="*50)

