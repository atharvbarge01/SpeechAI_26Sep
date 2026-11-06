"""
Configuration module for the 6-parameter Speech Scoring Model.
"""

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class ScorerConfig:
    # Audio Backbone
    backbone_name: str = "facebook/wav2vec2-base"
    sample_rate: int = 16000
    max_audio_length_seconds: float = 15.0
    
    # 6 Target Parameters
    parameter_names: List[str] = field(default_factory=lambda: [
        "wpm",            # Speech rate (Words Per Minute)
        "filler_count",   # Number of filler words detected
        "stammer_count",  # Number of stammers / repetitions
        "pause_count",    # Number of long pauses / gaps
        "clarity",        # Acoustic clarity / articulation score (0.0 - 10.0)
        "wer"             # Word Error Rate (0.0 - 1.0)
    ])
    
    num_parameters: int = 6
    
    # Regression Head Architecture
    hidden_dim: int = 256
    dropout: float = 0.2
    
    # Training Hyperparameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 30
    freeze_backbone_epochs: int = 5
    loss_type: str = "huber"
    
    # Multi-task Loss Weights
    parameter_weights: Dict[str, float] = field(default_factory=lambda: {
        "wpm": 1.0,
        "filler_count": 1.2,
        "stammer_count": 1.2,
        "pause_count": 1.0,
        "clarity": 1.5,
        "wer": 1.5
    })
    
    ideal_wpm_min: float = 120.0
    ideal_wpm_max: float = 150.0

