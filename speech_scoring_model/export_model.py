"""
Model Export and Optimization script.
Exports the trained model to TorchScript and ONNX formats
for high-throughput, low-latency deployment in production.
"""

import os
import sys
import argparse
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from speech_scoring_model.config import ScorerConfig
from speech_scoring_model.model import Wav2Vec2SpeechScorer

def export_model(
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_dir: str = "exported_models",
    export_torchscript: bool = True,
    export_onnx: bool = True
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", ScorerConfig())
    
    model = Wav2Vec2SpeechScorer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Dummy input: 5 seconds of 16kHz audio (80,000 samples)
    dummy_input = torch.randn(1, 80000, dtype=torch.float32)
    dummy_mask = torch.ones(1, 80000, dtype=torch.long)
    
    # 1. Export TorchScript Trace
    if export_torchscript:
        ts_path = os.path.join(output_dir, "speech_scorer_traced.pt")
        print(f"Exporting to TorchScript -> {ts_path}...")
        try:
            # Wrap model forward to return single tensor for simpler tracing
            class ScorerWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x, mask):
                    return self.m(x, mask)["predictions"]
                    
            wrapper = ScorerWrapper(model)
            traced_model = torch.jit.trace(wrapper, (dummy_input, dummy_mask))
            traced_model.save(ts_path)
            print(f" TorchScript export successful!")
        except Exception as e:
            print(f" TorchScript export failed: {e}")
            
    # 2. Export ONNX
    if export_onnx:
        onnx_path = os.path.join(output_dir, "speech_scorer.onnx")
        print(f"Exporting to ONNX -> {onnx_path}...")
        try:
            torch.onnx.export(
                wrapper if export_torchscript else model,
                (dummy_input, dummy_mask),
                onnx_path,
                input_names=["audio_waveform", "attention_mask"],
                output_names=["predictions"],
                dynamic_axes={
                    "audio_waveform": {0: "batch_size", 1: "audio_length"},
                    "attention_mask": {0: "batch_size", 1: "audio_length"},
                    "predictions": {0: "batch_size"}
                },
                opset_version=14
            )
            print(f" ONNX export successful!")
        except Exception as e:
            print(f" ONNX export failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Speech Scorer Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="exported_models")
    args = parser.parse_args()
    
    export_model(checkpoint_path=args.checkpoint, output_dir=args.output_dir)

