"""
Training pipeline for the 6-parameter Wav2Vec2 Speech Scoring Model.
Designed for execution on a separate PC or GPU environment (Colab/Kaggle/Dedicated workstation).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

# Ensure root is in python path
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

def evaluate(model, dataloader, scaler, config, device, loss_weights):
    model.eval()
    val_loss = 0.0
    all_preds_raw = []
    all_targets_raw = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            raw_targets = batch["raw_targets"].numpy()
            
            outputs = model(input_values, attention_mask=attention_mask)
            preds = outputs["predictions"]
            
            loss = model.compute_loss(preds, targets, weights=loss_weights)
            val_loss += loss.item() * len(input_values)
            
            # Denormalize predictions for physical error calculation
            preds_raw = scaler.inverse_transform(preds)
            all_preds_raw.append(preds_raw)
            all_targets_raw.append(raw_targets)
            
    val_loss /= len(dataloader.dataset)
    all_preds_raw = np.vstack(all_preds_raw)
    all_targets_raw = np.vstack(all_targets_raw)
    
    # Calculate MAE for each parameter
    maes = {}
    rmses = {}
    for i, name in enumerate(config.parameter_names):
        mae = float(np.mean(np.abs(all_preds_raw[:, i] - all_targets_raw[:, i])))
        rmse = float(np.sqrt(np.mean((all_preds_raw[:, i] - all_targets_raw[:, i]) ** 2)))
        maes[name] = round(mae, 3)
        rmses[name] = round(rmse, 3)
        
    return val_loss, maes, rmses


def train(args):
    config = ScorerConfig(
        backbone_name=args.backbone,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone_lr=args.backbone_lr,
        epochs=args.epochs,
        freeze_backbone_epochs=args.freeze_epochs
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"==================================================")
    print(f" Training 6-Parameter Speech Scorer")
    print(f" Backbone: {config.backbone_name}")
    print(f" Device:   {device}")
    print(f" Target Parameters: {config.parameter_names}")
    print(f"==================================================")
    
    # Check dataset existence
    metadata_path = os.path.join(args.data_dir, args.metadata_file)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        
    # Read metadata to split train/val
    full_df = pd.read_csv(metadata_path) if metadata_path.endswith(".csv") else pd.read_json(metadata_path)
    train_indices, val_indices = train_test_split(
        np.arange(len(full_df)),
        test_size=args.val_split,
        random_state=args.seed,
        shuffle=True
    )
    print(f"Dataset samples: Total={len(full_df)}, Train={len(train_indices)}, Val={len(val_indices)}")
    
    # Fit scaler on train set only to prevent leakage
    train_df = full_df.iloc[train_indices].copy()
    scaler = ParameterScaler(config.parameter_names)
    scaler.fit(train_df)
    scaler_save_path = os.path.join(args.output_dir, "scaler.json")
    scaler.save(scaler_save_path)
    print(f"Fitted target scaler and saved to {scaler_save_path}")
    
    # Create Datasets and DataLoaders
    train_dataset = SpeechScoringDataset(
        metadata_path=metadata_path,
        data_root_dir=args.data_dir,
        config=config,
        scaler=scaler,
        is_training=True
    )
    val_dataset = SpeechScoringDataset(
        metadata_path=metadata_path,
        data_root_dir=args.data_dir,
        config=config,
        scaler=scaler,
        is_training=False
    )
    
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_speech_batch,
        drop_last=False
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_speech_batch,
        drop_last=False
    )
    
    # Instantiate Model
    model = Wav2Vec2SpeechScorer(config)
    model.to(device)
    
    # Loss weights vector
    loss_weights = torch.tensor(
        [config.parameter_weights.get(p, 1.0) for p in config.parameter_names],
        dtype=torch.float32,
        device=device
    )
    
    # Optimizer with differential learning rates
    optimizer_params = [
        {"params": model.regression_head.parameters(), "lr": config.learning_rate},
        {"params": model.pooling.parameters(), "lr": config.learning_rate},
        {"params": model.backbone.parameters(), "lr": config.backbone_lr}
    ]
    optimizer = AdamW(optimizer_params, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    # Mixed precision scaler
    use_amp = args.fp16 and (device.type == "cuda")
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    history = {"train_loss": [], "val_loss": [], "maes": []}
    best_val_loss = float("inf")
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    
    print("\nStarting training loop...")
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        # Backbone warmup schedule
        if epoch <= config.freeze_backbone_epochs:
            model.freeze_backbone(True)
            freeze_status = "[Backbone Frozen]"
        else:
            model.freeze_backbone(False)
            model.freeze_feature_extractor()
            freeze_status = "[Full Fine-tuning]"
            
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_values, attention_mask=attention_mask)
                preds = outputs["predictions"]
                loss = model.compute_loss(preds, targets, weights=loss_weights)
                
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            train_loss += loss.item() * len(input_values)
            
        train_loss /= len(train_indices)
        scheduler.step()
        
        # Validation
        val_loss, val_maes, val_rmses = evaluate(model, val_loader, scaler, config, device, loss_weights)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["maes"].append(val_maes)
        
        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} {freeze_status} - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"WPM MAE: {val_maes['wpm']:.1f} | Filler MAE: {val_maes['filler_count']:.2f} | "
            f"WER MAE: {val_maes['wer']:.3f}"
        )
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "val_maes": val_maes,
                "val_rmses": val_rmses
            }, best_model_path)
            print(f"  --> Checkpoint saved: val_loss improved to {val_loss:.4f}")
            
    total_time = round(time.time() - start_time, 1)
    print(f"\nTraining completed in {total_time}s! Best model saved to: {best_model_path}")
    
    # Save training log and plot
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
        
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, config.epochs + 1), history["train_loss"], label="Train Loss")
        plt.plot(range(1, config.epochs + 1), history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Speech Scorer Training Curve")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(args.output_dir, "loss_curve.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Training loss curve plotted to: {plot_path}")
    except Exception as e:
        print(f"Notice: Could not save loss curve plot ({e})")
        
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 6-Parameter Speech Scorer Model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--metadata_file", type=str, default="metadata.csv", help="Metadata filename")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--backbone", type=str, default="facebook/wav2vec2-base", help="Pretrained Wav2Vec2 model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Regression head learning rate")
    parser.add_argument("--backbone_lr", type=float, default=2e-5, help="Backbone learning rate")
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Warmup epochs with frozen backbone")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation set split fraction")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train(args)

