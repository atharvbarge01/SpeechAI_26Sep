"""
Generates a small synthetic audio dataset in the `dataset/` folder
to verify the model architecture, dataloader, and training loop.
Uses only Python built-in libraries (wave, struct, math, csv, random).
"""

import os
import math
import struct
import wave
import csv
import random

def generate_sample_dataset(dataset_dir: str = "dataset", num_samples: int = 20):
    audio_dir = os.path.join(dataset_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    sr = 16000
    rows = []
    
    print(f"Generating {num_samples} sample audio WAV files in '{audio_dir}'...")
    
    for i in range(1, num_samples + 1):
        filename = f"sample_{i:03d}.wav"
        file_path = os.path.join(audio_dir, filename)
        
        # Duration between 3.0 and 6.0 seconds
        duration = random.uniform(3.0, 6.0)
        num_frames = int(sr * duration)
        
        base_freq = random.uniform(120.0, 240.0)
        pause_start = int(num_frames * 0.4)
        pause_end = pause_start + int(sr * random.uniform(0.5, 1.2))
        
        # Generate 16-bit PCM mono audio
        with wave.open(file_path, "w") as wav_file:
            wav_file.setnchannels(1)        # Mono
            wav_file.setsampwidth(2)       # 2 bytes = 16-bit
            wav_file.setframerate(sr)       # 16 kHz
            
            frames = bytearray()
            for frame_idx in range(num_frames):
                t = frame_idx / sr
                if pause_start <= frame_idx < pause_end:
                    # Silence gap (pause)
                    sample_val = 0
                else:
                    # Harmonic tones simulating voice
                    s = (
                        0.6 * math.sin(2 * math.pi * base_freq * t) +
                        0.3 * math.sin(2 * math.pi * (2 * base_freq) * t) +
                        0.1 * math.sin(2 * math.pi * (3 * base_freq) * t)
                    )
                    # Gentle amplitude envelope
                    sample_val = int(s * 24000)
                    sample_val = max(-32767, min(32767, sample_val))
                    
                frames.extend(struct.pack("<h", sample_val))
                
            wav_file.writeframes(frames)
            
        # Ground truth simulated metrics
        wpm = round(random.uniform(95.0, 155.0), 1)
        filler_count = random.randint(0, 5)
        stammer_count = random.randint(0, 3)
        pause_count = random.randint(1, 6)
        clarity = round(random.uniform(5.5, 9.6), 1)
        wer = round(random.uniform(0.02, 0.25), 3)
        
        rows.append({
            "audio_path": f"audio/{filename}",
            "wpm": wpm,
            "filler_count": filler_count,
            "stammer_count": stammer_count,
            "pause_count": pause_count,
            "clarity": clarity,
            "wer": wer
        })
        
    csv_path = os.path.join(dataset_dir, "metadata.csv")
    fieldnames = ["audio_path", "wpm", "filler_count", "stammer_count", "pause_count", "clarity", "wer"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Dataset generated successfully!")
    print(f"- Metadata written to: {csv_path}")
    print(f"- Audio samples count: {len(rows)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic sample dataset.")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--count", type=int, default=20, help="Number of samples")
    args = parser.parse_args()
    
    generate_sample_dataset(dataset_dir=args.output_dir, num_samples=args.count)
