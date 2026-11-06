"""
Deep Learning Model Architecture: Wav2Vec2 6-Parameter Speech Scorer.
Takes raw audio waveforms and directly predicts 6 speech therapy metrics:
1. WPM (Words Per Minute)
2. Filler word count
3. Stammer / repetition count
4. Long pause count
5. Acoustic clarity score (0-10)
6. Word Error Rate (WER: 0-1)
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional, Dict, Any

from speech_scoring_model.config import ScorerConfig

class AttentivePooling(nn.Module):
    """
    Learns an attention weight for each time-step frame in the audio sequence,
    allowing the model to focus on regions containing speech events (pauses, stammers, fillers).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden_states: (B, T, D)
        scores = self.attention(hidden_states)  # (B, T, 1)
        
        if attention_mask is not None:
            # Map input audio mask to downsampled feature time steps
            # Wav2Vec2 downsamples by ~320x
            T = hidden_states.size(1)
            downsampled_mask = nn.functional.interpolate(
                attention_mask.unsqueeze(1).float(),
                size=T,
                mode="nearest"
            ).squeeze(1).unsqueeze(-1)  # (B, T, 1)
            scores = scores.masked_fill(downsampled_mask == 0, -1e4)

        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        pooled = torch.sum(hidden_states * weights, dim=1)  # (B, D)
        return pooled


class Wav2Vec2SpeechScorer(nn.Module):
    def __init__(self, config: Optional[ScorerConfig] = None):
        super().__init__()
        self.config = config or ScorerConfig()
        
        # Load Wav2Vec2 backbone
        self.backbone = Wav2Vec2Model.from_pretrained(self.config.backbone_name)
        backbone_hidden_size = self.backbone.config.hidden_size
        
        # Attentive pooling layer
        self.pooling = AttentivePooling(backbone_hidden_size)
        
        # Multi-task Regression Head for 6 parameters
        self.regression_head = nn.Sequential(
            nn.Linear(backbone_hidden_size, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.LayerNorm(self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout / 2),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_parameters)
        )
        
        # Initialize head weights with small values
        for layer in self.regression_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def freeze_feature_extractor(self):
        """Freezes the raw convolutional feature encoder of Wav2Vec2 for faster training."""
        self.backbone.feature_extractor._freeze_parameters()

    def freeze_backbone(self, freeze: bool = True):
        """Freezes all layers of Wav2Vec2 backbone during head warmup."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_values: Raw normalized 16kHz audio waveforms of shape (Batch, AudioLength)
            attention_mask: Binary mask for padded frames of shape (Batch, AudioLength)
            
        Returns:
            Dictionary containing:
                - 'predictions': (Batch, 6) tensor of predicted normalized scores
                - 'pooled_features': (Batch, hidden_dim) acoustic embeddings
        """
        # Extract representations through Wav2Vec2
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, T, D)
        
        # Pool across time frames
        pooled = self.pooling(sequence_output, attention_mask=attention_mask)
        
        # Predict 6 parameters
        predictions = self.regression_head(pooled)
        
        return {
            "predictions": predictions,
            "pooled_features": pooled
        }

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes weighted Huber or MSE multi-task loss across the 6 parameters.
        """
        if self.config.loss_type == "huber":
            criterion = nn.SmoothL1Loss(reduction="none", beta=1.0)
        else:
            criterion = nn.MSELoss(reduction="none")
            
        loss_matrix = criterion(predictions, targets)  # (Batch, 6)
        
        if weights is not None:
            weights = weights.to(predictions.device)
            loss_matrix = loss_matrix * weights
            
        return loss_matrix.mean()

