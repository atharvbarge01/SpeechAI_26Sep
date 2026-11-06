"""
Inference module for the 6-parameter Speech Scoring Model inside backend.
"""

import os
import sys
import numpy as np
import librosa
import torch
from typing import Dict, Any, List, Optional

from ml_model.config import ScorerConfig
from ml_model.model import Wav2Vec2SpeechScorer
from ml_model.dataset import ParameterScaler

class SpeechScorerInference:
    def __init__(
        self,
        checkpoint_path: str,
        scaler_path: str,
        device: Optional[str] = None
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
        self.scaler = ParameterScaler.load(scaler_path)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint.get("config", ScorerConfig())
        self.model = Wav2Vec2SpeechScorer(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        max_samples = int(self.config.max_audio_length_seconds * self.config.sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        if np.max(np.abs(waveform)) > 1e-4:
            waveform = waveform / np.max(np.abs(waveform))
        return torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)

    def generate_feedback(self, metrics: Dict[str, float]) -> List[str]:
        suggestions = []
        wpm = metrics.get("wpm", 130.0)
        fillers = metrics.get("filler_count", 0.0)
        stammers = metrics.get("stammer_count", 0.0)
        pauses = metrics.get("pause_count", 0.0)
        clarity = metrics.get("clarity", 8.0)
        wer = metrics.get("wer", 0.05)

        if wpm < 115:
            suggestions.append(f"Speaking rate is cautious ({round(wpm, 1)} WPM). Focus on smooth continuity between phrases.")
        elif wpm > 155:
            suggestions.append(f"Pace is accelerated ({round(wpm, 1)} WPM). Insert deliberate pauses after key clauses.")
        else:
            suggestions.append(f"Great conversational rhythm ({round(wpm, 1)} WPM) in the optimal fluency range.")

        if fillers > 2:
            suggestions.append(f"Detected around {int(round(fillers))} filler pauses. Embrace silence instead of filler sounds.")
        if stammers > 1:
            suggestions.append(f"Noticed {int(round(stammers))} syllable repetitions. Use continuous phonation and relaxed jaw movements.")
        if pauses > 3:
            suggestions.append(f"Detected {int(round(pauses))} extended pauses. Chunk your thoughts into 4-6 word units.")
        if clarity < 6.5:
            suggestions.append(f"Articulation clarity is {round(clarity, 1)}/10. Enunciate terminal consonants distinctly.")
        elif wer > 0.15:
            suggestions.append(f"Pronunciation deviation is {round(wer*100, 1)}%. Practice minimal pair drills.")

        if not suggestions:
            suggestions.append("Exemplary speech clarity, steady pacing, and minimal disfluencies throughout.")

        return suggestions

    def predict(self, audio_path: str) -> Dict[str, Any]:
        input_tensor = self.preprocess_audio(audio_path).to(self.device)
        attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor, attention_mask=attention_mask)
            norm_predictions = outputs["predictions"].cpu().numpy()

        raw_preds = self.scaler.inverse_transform(norm_predictions)[0]
        
        metrics = {}
        for i, name in enumerate(self.config.parameter_names):
            metrics[name] = float(round(raw_preds[i], 2))

        wpm = metrics["wpm"]
        if 120.0 <= wpm <= 150.0:
            s_rate = 10.0
        elif wpm < 120.0:
            s_rate = max(0.0, 10.0 - ((120.0 - wpm) / 10.0))
        else:
            s_rate = max(0.0, 10.0 - ((wpm - 150.0) / 12.0))

        s_filler = max(0.0, min(10.0, 10.0 - (metrics["filler_count"] * 1.5)))
        s_stammer = max(0.0, min(10.0, 10.0 - (metrics["stammer_count"] * 2.0)))
        s_pause = max(0.0, min(10.0, 10.0 - (metrics["pause_count"] * 1.2)))
        s_clarity = metrics["clarity"]

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
            "model_source": "In-House Speech Scorer"
        }

