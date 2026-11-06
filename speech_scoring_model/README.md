# 🎯 6-Parameter Speech Scoring Model (In-House AI)

This model replaces external **Google Gemini API** calls with an in-house deep learning model trained directly on raw speech audio to predict the 6 clinical fluency parameters:

1. **`wpm`** — Speaking Rate (Words Per Minute)
2. **`filler_count`** — Vocalized Fillers (*um, uh, like*)
3. **`stammer_count`** — Repetitions, prolongations, and blocks
4. **`pause_count`** — Silent gaps & natural pauses
5. **`clarity`** — Articulatory acoustic clarity (0.0 to 10.0 scale)
6. **`wer`** — Word Error Rate / Disfluency deviation (0.0 to 1.0 scale)

---

## 📊 Dataset Analysis & Breakdown (37,625 Total Samples)

Your `dataset/` directory contains 3 major benchmark clinical stuttering and speech fluency datasets:

| Folder | Dataset Name | Audio Count | Annotation Type | Key Parameters Extracted |
|---|---|---|---|---|
| **`archive (8)`** | **UCLASS** (Univ. College London) | 4,712 WAV clips (3s, 16kHz) | `clips/labels.csv` | Blocks, Prolongations, Sound Reps, Word Reps, Interjections |
| **`archive (9)`** | **LibriStutter** | 4,736 FLAC files | 4,736 CSV time-aligned files + Transcripts | Word timestamps, exact duration, injected stutter types |
| **`archive (10)`** | **SEP-28k & FluencyBank** (Apple Research) | 28,177 WAV clips (3s, 16kHz) | `SEP-28k_labels.csv` | Multi-annotator votes: Blocks, Prolongations, Sound/Word Reps, Interjections, Natural Pauses, Audio Quality |
| **TOTAL** | **Unified Corpus** | **37,625 Samples** | Standardized CSV | All 6 Clinical Parameters |

---

## ⚙️ How Parameters Are Mapped from the Datasets

1. **`wpm`**:
   - For 3.0-second clips (SEP-28k & UCLASS), computed based on effective fluent word throughput scaled to 60s.
   - For LibriStutter, computed from exact word count divided by utterance duration.
2. **`filler_count`**:
   - Mapped from `Interjection` annotator counts in SEP-28k & UCLASS.
3. **`stammer_count`**:
   - Mapped from `Block + Prolongation + SoundRep + WordRep` across all datasets.
4. **`pause_count`**:
   - Mapped from `NaturalPause` in SEP-28k, silence blocks in UCLASS, and inter-word silence intervals (>0.6s) in LibriStutter.
5. **`clarity`**:
   - 0.0 to 10.0 score based on speech intelligibility, penalized by `PoorAudioQuality`, `DifficultToUnderstand`, and disfluency severity.
6. **`wer`**:
   - Phonetic and transcript deviations caused by blocks, repetitions, and vocalized intrusions.

---

## 🛠️ Step 1: Unify & Prepare Metadata

To generate the unified `dataset/metadata.csv`, run:

```bash
# Option A: Full dataset (37,625 samples)
python speech_scoring_model/prepare_unified_dataset.py --dataset_dir dataset --output_file metadata.csv

# Option B: Balanced subset for faster training (e.g., 10,000 samples)
python speech_scoring_model/prepare_unified_dataset.py --dataset_dir dataset --output_file metadata.csv --max_samples 10000
```

---

## 🚀 Step 2: Training on Your Separate PC

### Option A: Local PC / Dedicated Workstation (NVIDIA GPU)
1. Copy `dataset/` and `speech_scoring_model/` to your separate PC.
2. Install dependencies:
   ```bash
   pip install -r speech_scoring_model/requirements.txt
   ```
3. Run training:
   ```bash
   python speech_scoring_model/train.py \
       --data_dir dataset \
       --metadata_file metadata.csv \
       --output_dir checkpoints \
       --epochs 25 \
       --batch_size 16 \
       --device cuda \
       --fp16
   ```
4. Evaluate:
   ```bash
   python speech_scoring_model/evaluate.py \
       --data_dir dataset \
       --metadata_file metadata.csv \
       --checkpoint checkpoints/best_model.pt \
       --scaler checkpoints/scaler.json \
       --output_dir evaluation_reports
   ```

### Option B: Google Colab / Kaggle (Cloud GPU)
1. Open Google Colab and upload `speech_scoring_model/train_on_colab_or_kaggle.ipynb`.
2. Zip and upload your `dataset` folder.
3. Run all cells to train with free/Pro GPU acceleration and download `best_model.pt` + `scaler.json`.

---

## 🔌 Step 3: Connect Model to SpeechAI Project
When training completes:
1. Copy `best_model.pt` and `scaler.json` into:
   ```
   final_year_project/checkpoints/
   ```
2. The backend (`SpeechAI_26Sep/backend/main.py`) automatically prioritizes your in-house trained model for speech fluency scoring and clinical recommendations!
