"""
Dataset Unification & Preprocessing Engine for SpeechAI.
Combines 3 Benchmark Stuttering & Speech Fluency Datasets:
  1. archive (8)  - UCLASS (4,712 clips)
  2. archive (9)  - LibriStutter (4,736 full audio & annotations)
  3. archive (10) - SEP-28k & FluencyBank (28,177 clips)

Transforms raw disfluency annotations into continuous ground-truth values
for the 6 Clinical Fluency Parameters:
  1. wpm           - Speaking rate (Words per Minute)
  2. filler_count  - Vocalized fillers / interjections
  3. stammer_count - Repetitions, prolongations & blocks
  4. pause_count   - Gaps and natural pauses
  5. clarity       - Articulatory acoustic clarity (0.0 to 10.0)
  6. wer           - Word Error Rate / Disfluency deviation (0.0 to 1.0)
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import soundfile as sf

def process_archive_8(archive_dir: str) -> list[dict]:
    """Processes UCLASS dataset (archive 8)."""
    labels_csv = os.path.join(archive_dir, "clips", "labels.csv")
    if not os.path.exists(labels_csv):
        print(f"Notice: {labels_csv} not found, skipping archive 8.")
        return []
        
    df = pd.read_csv(labels_csv)
    records = []
    clips_dir = os.path.join(archive_dir, "clips")
    
    print(f"Processing Archive 8 (UCLASS): {len(df)} samples...")
    for _, row in df.iterrows():
        rel_path = str(row["filepath"]).replace("\\", "/")
        full_audio_path = os.path.join(clips_dir, rel_path)
        
        # In case filepath starts with clips/
        if not os.path.exists(full_audio_path):
            full_audio_path = os.path.join(archive_dir, rel_path)
            
        if not os.path.exists(full_audio_path):
            continue
            
        block = int(row.get("Block", 0))
        prolongation = int(row.get("Prolongation", 0))
        sound_rep = int(row.get("SoundRep", 0))
        word_rep = int(row.get("WordRep", 0))
        interjection = int(row.get("Interjection", 0))
        no_stutter = int(row.get("NoStutteredWords", 0))
        
        stammer_count = block + prolongation + sound_rep + word_rep
        filler_count = interjection
        
        # Each clip is exactly 3.0 seconds long
        # Estimate words: typical 3s clip has ~6 words fluent, reduced by stammer events
        est_words = max(1.0, 6.5 - (stammer_count * 1.5) - (filler_count * 0.5))
        wpm = round(float(est_words / (3.0 / 60.0)), 1)
        
        pause_count = 1 if block > 0 else 0
        
        # Clarity: penalized by disfluencies (10.0 scale)
        clarity = round(float(np.clip(9.5 - (stammer_count * 1.8) - (filler_count * 0.8), 2.5, 9.8)), 1)
        
        # WER estimation: ratio of disfluent events over total words
        wer = round(float(np.clip(stammer_count / max(1.0, est_words), 0.0, 0.95)), 3)
        
        # Store path relative to dataset root
        dataset_rel_path = os.path.relpath(full_audio_path, os.path.dirname(archive_dir)).replace("\\", "/")
        
        records.append({
            "audio_path": dataset_rel_path,
            "wpm": wpm,
            "filler_count": filler_count,
            "stammer_count": stammer_count,
            "pause_count": pause_count,
            "clarity": clarity,
            "wer": wer,
            "source_dataset": "UCLASS"
        })
        
    print(f"Archive 8 processed: {len(records)} verified records.")
    return records


def process_archive_10(archive_dir: str) -> list[dict]:
    """Processes SEP-28k / FluencyBank dataset (archive 10)."""
    labels_csv = os.path.join(archive_dir, "SEP-28k_labels.csv")
    clips_dir = os.path.join(archive_dir, "clips", "stuttering-clips", "clips")
    
    if not os.path.exists(labels_csv) or not os.path.exists(clips_dir):
        print(f"Notice: {labels_csv} or {clips_dir} not found, skipping archive 10.")
        return []
        
    df = pd.read_csv(labels_csv)
    records = []
    
    print(f"Processing Archive 10 (SEP-28k): {len(df)} samples...")
    for _, row in df.iterrows():
        show = str(row["Show"]).strip()
        ep = str(row["EpId"]).strip()
        clip = str(row["ClipId"]).strip()
        filename = f"{show}_{ep}_{clip}.wav"
        full_audio_path = os.path.join(clips_dir, filename)
        
        if not os.path.exists(full_audio_path):
            continue
            
        # Extract annotator votes (range 0 to 3)
        block = int(row.get("Block", 0))
        prolongation = int(row.get("Prolongation", 0))
        sound_rep = int(row.get("SoundRep", 0))
        word_rep = int(row.get("WordRep", 0))
        interjection = int(row.get("Interjection", 0))
        natural_pause = int(row.get("NaturalPause", 0))
        poor_quality = int(row.get("PoorAudioQuality", 0))
        difficult = int(row.get("DifficultToUnderstand", 0))
        
        stammer_count = int(round((block + prolongation + sound_rep + word_rep) / 2.0))
        filler_count = interjection
        pause_count = natural_pause
        
        # 3.0s clips
        est_words = max(1.0, 7.0 - (stammer_count * 1.4) - (filler_count * 0.6))
        wpm = round(float(est_words / (3.0 / 60.0)), 1)
        
        # Clarity (0-10)
        clarity_score = 9.8 - (poor_quality * 2.0) - (difficult * 2.5) - (stammer_count * 1.2)
        clarity = round(float(np.clip(clarity_score, 1.5, 9.8)), 1)
        
        # WER
        wer = round(float(np.clip((stammer_count + filler_count * 0.5) / max(1.0, est_words), 0.0, 0.90)), 3)
        
        dataset_rel_path = os.path.relpath(full_audio_path, os.path.dirname(archive_dir)).replace("\\", "/")
        
        records.append({
            "audio_path": dataset_rel_path,
            "wpm": wpm,
            "filler_count": filler_count,
            "stammer_count": stammer_count,
            "pause_count": pause_count,
            "clarity": clarity,
            "wer": wer,
            "source_dataset": "SEP-28k"
        })
        
    print(f"Archive 10 processed: {len(records)} verified records.")
    return records


def process_archive_9(archive_dir: str, max_samples: int = 5000) -> list[dict]:
    """Processes LibriStutter dataset (archive 9)."""
    audio_dir = os.path.join(archive_dir, "LibriStutter Audio")
    ann_dir = os.path.join(archive_dir, "LibriStutter Annotations")
    
    if not os.path.exists(audio_dir) or not os.path.exists(ann_dir):
        print(f"Notice: {audio_dir} or {ann_dir} not found, skipping archive 9.")
        return []
        
    records = []
    print(f"Processing Archive 9 (LibriStutter)...")
    
    csv_files = []
    for root, _, files in os.walk(ann_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
                
    if max_samples and len(csv_files) > max_samples:
        np.random.seed(42)
        csv_files = list(np.random.choice(csv_files, size=max_samples, replace=False))
        
    for ann_file in csv_files:
        rel = os.path.relpath(ann_file, ann_dir)
        audio_rel = os.path.splitext(rel)[0] + ".flac"
        full_audio_path = os.path.join(audio_dir, audio_rel)
        
        if not os.path.exists(full_audio_path):
            continue
            
        try:
            # Parse annotation: word, start, end, label
            df_ann = pd.read_csv(ann_file, header=None, names=["word", "start", "end", "label"])
            if len(df_ann) == 0:
                continue
                
            total_duration = float(df_ann["end"].max()) - float(df_ann["start"].min())
            if total_duration <= 0.5:
                continue
                
            # Words count (excluding STUTTER tokens)
            actual_words = df_ann[df_ann["word"] != "STUTTER"]
            num_words = len(actual_words)
            
            # Stutter events (label > 0 or word == "STUTTER")
            stutters = df_ann[(df_ann["label"] > 0) | (df_ann["word"] == "STUTTER")]
            stammer_count = len(stutters)
            
            # WPM
            duration_min = total_duration / 60.0
            wpm = round(float(num_words / duration_min), 1) if duration_min > 0 else 120.0
            
            # Gaps between consecutive words > 0.6s
            pause_count = 0
            df_sorted = df_ann.sort_values(by="start")
            starts = df_sorted["start"].values
            ends = df_sorted["end"].values
            for i in range(len(starts) - 1):
                if starts[i+1] - ends[i] > 0.6:
                    pause_count += 1
                    
            filler_count = 0  # LibriStutter is predominantly read speech with stutters
            
            clarity = round(float(np.clip(9.6 - (stammer_count * 1.5), 3.0, 9.8)), 1)
            wer = round(float(np.clip(stammer_count / max(1.0, num_words), 0.0, 0.85)), 3)
            
            dataset_rel_path = os.path.relpath(full_audio_path, os.path.dirname(archive_dir)).replace("\\", "/")
            
            records.append({
                "audio_path": dataset_rel_path,
                "wpm": wpm,
                "filler_count": filler_count,
                "stammer_count": stammer_count,
                "pause_count": pause_count,
                "clarity": clarity,
                "wer": wer,
                "source_dataset": "LibriStutter"
            })
        except Exception:
            continue
            
    print(f"Archive 9 processed: {len(records)} verified records.")
    return records


def build_unified_dataset(dataset_dir: str, output_csv: str = "metadata.csv", balance_samples: int = None):
    arch8 = os.path.join(dataset_dir, "archive (8)")
    arch9 = os.path.join(dataset_dir, "archive (9)")
    arch10 = os.path.join(dataset_dir, "archive (10)")
    
    rec8 = process_archive_8(arch8)
    rec9 = process_archive_9(arch9)
    rec10 = process_archive_10(arch10)
    
    all_records = rec8 + rec9 + rec10
    if not all_records:
        print("Error: No records found across archive folders!")
        return
        
    valid_records = []
    skipped_empty_audio = 0
    for record in all_records:
        audio_path = os.path.join(dataset_dir, record["audio_path"])
        if sf.info(audio_path).frames == 0:
            skipped_empty_audio += 1
            continue
        valid_records.append(record)

    if skipped_empty_audio:
        print(f"Skipped {skipped_empty_audio} records with empty audio files.")

    all_records = valid_records
    if not all_records:
        print("Error: No records with non-empty audio found across archive folders!")
        return

    df = pd.DataFrame(all_records)
    print("\n" + "="*60)
    print(f" UNIFIED DATASET SUMMARY: {len(df)} TOTAL SAMPLES")
    print("="*60)
    print(df["source_dataset"].value_counts())
    
    if balance_samples and balance_samples < len(df):
        print(f"\nSubsampling to balanced {balance_samples} instances for faster training...")
        df = df.sample(n=balance_samples, random_state=42).reset_index(drop=True)
        
    target_csv = os.path.join(dataset_dir, output_csv)
    df.to_csv(target_csv, index=False)
    print(f"\nSaved unified metadata to: {target_csv}")
    
    print("\nDataset Parameter Statistics:")
    print(df[["wpm", "filler_count", "stammer_count", "pause_count", "clarity", "wer"]].describe().round(2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unify 3 archive datasets into metadata.csv")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Root dataset directory")
    parser.add_argument("--output_file", type=str, default="metadata.csv", help="Output CSV filename")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max sample limit (e.g. 10000)")
    args = parser.parse_args()
    
    build_unified_dataset(
        dataset_dir=args.dataset_dir,
        output_csv=args.output_file,
        balance_samples=args.max_samples
    )
