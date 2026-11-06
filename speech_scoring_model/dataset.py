"""
Dataset and Data Loading pipeline for the 6-parameter Speech Scoring Model.
Supports loading audio files (.wav, .mp3, .ogg, .flac) paired with metadata (.csv, .json).
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple, Union

from speech_scoring_model.config import ScorerConfig

class ParameterScaler:
    """
    Normalizes the 6 target parameters to zero-mean and unit-variance
    (or min-max) to ensure balanced multi-task regression gradients.
    """
    def __init__(self, parameter_names: List[str]):
        self.parameter_names = parameter_names
        self.means = {k: 0.0 for k in parameter_names}
        self.stds = {k: 1.0 for k in parameter_names}
        self.mins = {k: 0.0 for k in parameter_names}
        self.maxs = {k: 1.0 for k in parameter_names}
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        for col in self.parameter_names:
            if col in df.columns:
                vals = df[col].astype(float).values
                self.means[col] = float(np.mean(vals))
                std = float(np.std(vals))
                self.stds[col] = std if std > 1e-6 else 1.0
                self.mins[col] = float(np.min(vals))
                self.maxs[col] = float(np.max(vals))
        self.fitted = True

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Standardizes a (..., 6) numpy array or list."""
        targets = np.array(targets, dtype=np.float32)
        norm = np.zeros_like(targets)
        for i, col in enumerate(self.parameter_names):
            norm[..., i] = (targets[..., i] - self.means[col]) / self.stds[col]
        return norm

    def inverse_transform(self, normalized_targets: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Denormalizes back to original physical units (WPM, counts, etc.)."""
        if isinstance(normalized_targets, torch.Tensor):
            normalized_targets = normalized_targets.detach().cpu().numpy()
        normalized_targets = np.array(normalized_targets, dtype=np.float32)
        original = np.zeros_like(normalized_targets)
        for i, col in enumerate(self.parameter_names):
            val = normalized_targets[..., i] * self.stds[col] + self.means[col]
            # Clip physical minimums (counts, WER >= 0)
            if col in ["filler_count", "stammer_count", "pause_count", "wpm", "clarity", "wer"]:
                val = np.maximum(0.0, val)
            if col == "clarity":
                val = np.minimum(10.0, val)
            if col == "wer":
                val = np.minimum(1.0, val)
            original[..., i] = val
        return original

    def to_dict(self) -> Dict:
        return {
            "parameter_names": self.parameter_names,
            "means": self.means,
            "stds": self.stds,
            "mins": self.mins,
            "maxs": self.maxs,
            "fitted": self.fitted
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ParameterScaler":
        with open(filepath, "r") as f:
            data = json.load(f)
        scaler = cls(data["parameter_names"])
        scaler.means = data["means"]
        scaler.stds = data["stds"]
        scaler.mins = data.get("mins", {})
        scaler.maxs = data.get("maxs", {})
        scaler.fitted = data.get("fitted", True)
        return scaler


class SpeechScoringDataset(Dataset):
    """
    PyTorch Dataset that pairs audio files with the 6 parameters.
    """
    def __init__(
        self,
        metadata_path: str,
        data_root_dir: str,
        config: Optional[ScorerConfig] = None,
        scaler: Optional[ParameterScaler] = None,
        is_training: bool = True
    ):
        self.config = config or ScorerConfig()
        self.data_root_dir = os.path.abspath(data_root_dir)
        self.is_training = is_training
        
        # Load metadata (CSV or JSON)
        if metadata_path.endswith(".json"):
            self.df = pd.read_json(metadata_path)
        else:
            self.df = pd.read_csv(metadata_path)
            
        # Ensure all 6 target parameters exist in dataframe
        for param in self.config.parameter_names:
            if param not in self.df.columns:
                raise ValueError(f"Required parameter column '{param}' missing in metadata.")
                
        # Setup scaler
        if scaler is None:
            self.scaler = ParameterScaler(self.config.parameter_names)
            self.scaler.fit(self.df)
        else:
            self.scaler = scaler

    def __len__(self) -> int:
        return len(self.df)

    def load_audio(self, audio_rel_path: str) -> np.ndarray:
        """Loads and resamples audio to target sample rate (16000 Hz mono)."""
        full_path = os.path.join(self.data_root_dir, audio_rel_path)
        if not os.path.exists(full_path):
            # Also check if path was already absolute or relative to cwd
            if os.path.exists(audio_rel_path):
                full_path = audio_rel_path
            else:
                raise FileNotFoundError(f"Audio file not found: {full_path}")
                
        # Read audio
        waveform, sr = librosa.load(full_path, sr=self.config.sample_rate, mono=True)
        
        # Truncate if longer than max audio length
        max_samples = int(self.config.max_audio_length_seconds * self.config.sample_rate)
        if len(waveform) > max_samples:
            if self.is_training:
                # Random crop during training for data augmentation
                max_start = len(waveform) - max_samples
                start = np.random.randint(0, max_start)
                waveform = waveform[start : start + max_samples]
            else:
                waveform = waveform[:max_samples]
                
        # Amplitude normalization
        if np.max(np.abs(waveform)) > 1e-4:
            waveform = waveform / np.max(np.abs(waveform))
            
        return waveform.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        audio_path = str(row["audio_path"])
        waveform = self.load_audio(audio_path)
        
        # Extract raw targets
        raw_targets = np.array([float(row[p]) for p in self.config.parameter_names], dtype=np.float32)
        
        # Standardize targets
        normalized_targets = self.scaler.transform(raw_targets)
        
        return {
            "waveform": torch.from_numpy(waveform),
            "targets": torch.from_numpy(normalized_targets),
            "raw_targets": torch.from_numpy(raw_targets),
            "audio_path": audio_path
        }


def collate_speech_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Pads variable length audio waveforms to the max duration in the batch
    and generates corresponding binary attention masks.
    """
    waveforms = [item["waveform"] for item in batch]
    targets = torch.stack([item["targets"] for item in batch], dim=0)
    raw_targets = torch.stack([item["raw_targets"] for item in batch], dim=0)
    audio_paths = [item["audio_path"] for item in batch]
    
    lengths = [len(w) for w in waveforms]
    max_len = max(lengths)
    
    padded_waveforms = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
    attention_masks = torch.zeros(len(waveforms), max_len, dtype=torch.long)
    
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :len(w)] = w
        attention_masks[i, :len(w)] = 1
        
    return {
        "input_values": padded_waveforms,
        "attention_mask": attention_masks,
        "targets": targets,
        "raw_targets": raw_targets,
        "audio_paths": audio_paths
    }

