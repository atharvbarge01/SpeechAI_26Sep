"""
Model Evaluation and Benchmarking Script.
Calculates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
and Pearson Correlation Coefficient (r) across all 6 parameters on test data.
Produces publication-quality metric reports and correlation plots.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from speech_scoring_model.config import ScorerConfig
from speech_scoring_model.model import Wav2Vec2SpeechScorer
from speech_scoring_model.dataset import (
    SpeechScoringDataset,
    ParameterScaler,
    collate_speech_batch
)

def run_evaluation(
    data_dir: str,
    metadata_file: str,
    checkpoint_path: str,
    scaler_path: str,
    output_dir: str = "evaluation_reports",
    device: str = "cpu"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", ScorerConfig())
    
    scaler = ParameterScaler.load(scaler_path)
    model = Wav2Vec2SpeechScorer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    metadata_path = os.path.join(data_dir, metadata_file)
    dataset = SpeechScoringDataset(
        metadata_path=metadata_path,
        data_root_dir=data_dir,
        config=config,
        scaler=scaler,
        is_training=False
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_speech_batch)
    
    all_preds = []
    all_targets = []
    
    print(f"Evaluating {len(dataset)} samples...")
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_values, attention_mask=attention_mask)
            
            raw_targets = batch["raw_targets"].numpy()
            raw_preds = scaler.inverse_transform(outputs["predictions"])
            
            all_preds.append(raw_preds)
            all_targets.append(raw_targets)
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    metrics_summary = []
    
    print("\n" + "="*70)
    print(f"{'PARAMETER':<18} | {'MAE':<10} | {'RMSE':<10} | {'PEARSON r':<12} | {'R² SCORE':<10}")
    print("="*70)
    
    for i, name in enumerate(config.parameter_names):
        true_vals = all_targets[:, i]
        pred_vals = all_preds[:, i]
        
        mae = float(np.mean(np.abs(pred_vals - true_vals)))
        rmse = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))
        
        # Pearson r
        if np.std(true_vals) > 1e-5 and np.std(pred_vals) > 1e-5:
            r, _ = pearsonr(true_vals, pred_vals)
        else:
            r = 0.0
            
        r2 = r2_score(true_vals, pred_vals)
        
        metrics_summary.append({
            "parameter": name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "pearson_r": round(float(r), 4),
            "r2": round(float(r2), 4)
        })
        
        print(f"{name:<18} | {mae:<10.3f} | {rmse:<10.3f} | {r:<12.3f} | {r2:<10.3f}")
        
    print("="*70)
    
    # Save CSV report
    report_df = pd.DataFrame(metrics_summary)
    report_csv = os.path.join(output_dir, "evaluation_metrics.csv")
    report_df.to_csv(report_csv, index=False)
    print(f"Metrics saved to: {report_csv}")
    
    # Generate 6-panel correlation plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(config.parameter_names):
        ax = axes[i]
        true_vals = all_targets[:, i]
        pred_vals = all_preds[:, i]
        
        ax.scatter(true_vals, pred_vals, alpha=0.7, edgecolors="k", s=40)
        
        # Identity line
        min_v = min(np.min(true_vals), np.min(pred_vals))
        max_v = max(np.max(true_vals), np.max(pred_vals))
        ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.5, label="Perfect Alignment")
        
        ax.set_title(f"{name.upper()} (r = {metrics_summary[i]['pearson_r']:.2f})")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Model Prediction")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "prediction_correlation.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Correlation plot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained speech scorer")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--metadata_file", type=str, default="metadata.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--scaler", type=str, default="checkpoints/scaler.json")
    parser.add_argument("--output_dir", type=str, default="evaluation_reports")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    run_evaluation(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        output_dir=args.output_dir,
        device=args.device
    )

