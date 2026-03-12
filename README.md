# okyeame-tts

> **Okyeame** (ɔkyeame) — the royal linguist and spokesperson in Ghanaian tradition, renowned for clarity, eloquence, and the power of voice.

A fine-tuned XTTS v2 text-to-speech model that speaks English with a natural Ghanaian accent. Built on the `ghananlpcommunity/ghana-english-asr-2700hrs` dataset — 30,000+ training chunks drawn exclusively from Ghanaian English speech — okyeame-tts is the first open-source TTS model **fine-tuned entirely and specifically on Ghanaian English data**.

---

## Why This Exists

Every major TTS system — ElevenLabs, Google, Amazon Polly — speaks with a Western accent by default. Ghanaian English is a distinct, legitimate variety of English spoken by millions, yet it has little representation in open-source TTS.

Some prior work exists. [Afro-TTS](https://huggingface.co/intronhealth/afro-tts) (`intronhealth/afro-tts`) is a pan-African accented English TTS system that covers 86 African accents including Ghanaian — a significant achievement. [Abena AI](https://abena.ai) offers a closed-source Ghanaian English TTS product. Both are valuable contributions to the space.

okyeame-tts takes a different approach: **depth over breadth**. Rather than covering many African accents broadly, it trains exclusively on Ghanaian English speech — 30,000+ aligned audio chunks from the Ghana English ASR dataset — optimizing specifically for the phonetic and prosodic patterns of Ghanaian English. The Ghanaian accent is the native voice of the model, not one accent among many.

As a zero-shot voice cloning model, it can clone any Ghanaian voice from a short reference audio clip — giving access to thousands of distinct Ghanaian voices from the training corpus.

---

## Project Status

| Stage | Status | Description |
|---|---|---|
| Data Collection | ✅ Done | 10,000 samples, 37hrs from ASR dataset |
| Forced Alignment | ✅ Done | 30,536 chunks, 29.98 hours, avg 3.5s |
| Coqui Format | ✅ Done | 27,038 train / 3,005 eval samples at 22050Hz |
| XTTS v2 Fine-tuning | ✅ Done | Stage 1 complete — model sounds Ghanaian |
| Evaluation (UTMOS) | ✅ Done | Top-20 reference speakers scored and saved |
| Evaluation (WER + MOS) | ⏳ Stage 2 | Formal intelligibility and human evaluation |
| HuggingFace Release | ⏳ Pending | Stage 2 (50k-100k samples) |

---

## Model Architecture

XTTS v2 is a multi-stage pipeline. Fine-tuning targets the GPT-2 decoder only — the component that learns the mapping from text to audio codes conditioned on a speaker embedding.

```
TEXT INPUT
    ↓
[Tokenizer]           — converts text to token IDs (vocab.json)
    ↓
[GPT-2 Decoder] ◄─── [Speaker Encoder] ◄─── reference wav
    │                   extracts 512-dim
    │                   speaker embedding
    ↓
[DVAE]                — encodes reference audio into discrete codebook tokens
    ↓
[HiFi-GAN Vocoder]    — converts predicted audio codes → waveform (24000Hz)
    ↓
AUDIO OUTPUT
```

**What fine-tuning changes:** The GPT-2 decoder learns Ghanaian English phonetic and prosodic patterns from the training data. The speaker encoder and vocoder are frozen — voice cloning capability and audio quality are preserved.

**Why eval loss plateaus:** The decoder overfits after ~1 epoch on the Stage 1 subset. Eval loss stopped improving at 3.527. Training loss continued decreasing. Stage 2 addresses this by training on a larger, more diverse subset of the full 2,700-hour dataset.

---

## Repository Structure

```
okyeame-tts/
├── config.yaml                    # Project-wide configuration and defaults
├── requirements.txt               # Python dependencies
├── data/
│   └── processing/
│       ├── 01_collect.py          # Collect and filter samples from ASR dataset
│       ├── 02_align.py            # Forced alignment — text to audio chunks
│       └── 03_format.py           # Format chunks into Coqui/LJSpeech structure
├── training/
│   └── train.py                   # Fine-tune XTTS v2 on formatted dataset
├── evaluation/
│   └── evaluate.py                # UTMOS reference speaker scoring
└── notebooks/
    └── exploration.ipynb          # Exploratory data analysis
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/derricknyarko/okyeame-tts.git
cd okyeame-tts
pip install -r requirements.txt
```

> **Note:** `requirements.txt` uses `git+https://github.com/idiap/coqui-ai-TTS.git` — the community-maintained Coqui TTS fork — because the original `TTS` package does not support Python 3.12.

### 2. Apply the Monkey Patch (Required)

Due to a version incompatibility between `coqui-ai-TTS` and newer versions of `transformers`, a missing function must be patched before importing TTS modules. Run this **once per session** before any TTS import:

```python
import transformers.pytorch_utils as pt_utils
import torch

def isin_mps_friendly(elements, test_elements):
    return torch.isin(elements, test_elements)

pt_utils.isin_mps_friendly = isin_mps_friendly
```

> **Root cause:** `isin_mps_friendly` was removed from `transformers.pytorch_utils` in newer versions of `transformers`. The coqui-ai-TTS fork still imports it. The fix re-adds the function using `torch.isin`, which is the underlying implementation.

### 3. Run Inference

```python
import torch
import numpy as np
import scipy.io.wavfile as wav_io
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Load model directly using XTTS classes
config = XttsConfig()
config.load_json("path/to/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path="path/to/best_model_9013.pth", eval=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# vocab.json must be in the same directory as config.json
# Download from: https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json

# Generate speech — single sentence
outputs = model.synthesize(
    "The government has announced new policies to support local businesses in Ghana.",
    config,
    speaker_wav="path/to/reference_speaker.wav",
    gpt_cond_len=6,
    language="en",
)
wav_io.write("output.wav", config.audio.output_sample_rate, outputs["wav"])
```

> **Reference audio:** Any clean wav file from the training corpus can be used as a speaker reference. The model clones the voice characteristics while preserving the Ghanaian accent. UTMOS scoring (see Evaluation) can be used to identify the highest quality reference speakers.

> **Long text:** For best quality, split text at sentence boundaries and concatenate outputs with a short silence between them rather than passing the full passage at once.

---

## Configuration

All paths and dataset parameters are set in `config.yaml` at the project root.

```yaml
paths:
  metadata: '/kaggle/working/metadata.pkl'
  audio_dir: '/kaggle/working/audio'
  output: '/kaggle/working'
  ljspeech_output: '/kaggle/working/ljspeech'

dataset:
  name: 'ghananlpcommunity/ghana-english-asr-2700hrs'
  target: 10000
```

---

## Pipeline

```
01_collect.py  →  metadata.pkl + audio/*.wav
      ↓
02_align.py    →  aligned_metadata.pkl + aligned_audio/*.wav
      ↓
03_format.py   →  wavs/*.wav + metadata_train.csv + metadata_eval.csv
      ↓
train.py       →  best_model_XXXX.pth (fine-tuned XTTS v2)
      ↓
evaluate.py    →  utmos_scores.csv + top reference speakers
```

---

### Step 1 — Data Collection

Streams the `ghananlpcommunity/ghana-english-asr-2700hrs` dataset and filters for quality samples.

**Filters applied:**
- SNR > 2
- Duration: 2–16 seconds
- Text length: 20–300 characters

```bash
python data/processing/01_collect.py \
  --dataset_name ghananlpcommunity/ghana-english-asr-2700hrs \
  --output_dir /kaggle/working \
  --target 10000
```

**Output:**
```
output_dir/
├── metadata.pkl
└── audio/
    ├── 00000.wav
    └── ...
```

**Stage 1 result:** 10,000 samples, 37.36 hours, 16000Hz

---

### Step 2 — Forced Alignment

Uses `ctc-forced-aligner` to align transcripts to audio at the word level, then splits audio into natural chunks at pauses and punctuation boundaries.

```bash
python data/processing/02_align.py \
  --metadata_path /kaggle/working/metadata.pkl \
  --audio_dir /kaggle/working/audio \
  --output_dir /kaggle/working \
  --target 10000
```

**Checkpointing:** Saves every 100 samples to `aligned_metadata.pkl`. Interrupted runs resume automatically.

**Output:**
```
output_dir/
├── aligned_metadata.pkl
└── aligned_audio/
    ├── 00000.wav        # 2–7 second aligned chunks at 16000Hz
    └── ...
```

**Stage 1 result:** 30,536 chunks, 0 failures, 29.98 hours, avg 3.5s per chunk

**Expected runtime (Kaggle P100):** ~3.8s per sample → ~10 hours for 10,000 samples

---

### Step 3 — Coqui Format

Converts aligned chunks into Coqui CSV format. Resamples audio from 16000Hz to 22050Hz. Splits into train/eval.

```bash
python data/processing/03_format.py \
  --input_dir /kaggle/working \
  --audio_dir /kaggle/input/datasets/derricknyarko/okyeame-aligned-chunks-v2/aligned_audio \
  --output_dir /kaggle/working/coqui_format \
  --eval_split 0.1
```

> **Note:** Use `--audio_dir` to remap audio paths when loading from a Kaggle input dataset across sessions.

**Filters applied:**
- Skips chunks with empty text
- Skips chunks with text longer than 150 characters
- Skips chunks with more than 1 filler word (uh, um, hmm, hm, ah)

**Output:**
```
output_dir/
├── wavs/
├── metadata_train.csv
└── metadata_eval.csv
```

**CSV format:**
```
audio_file|text|speaker_name
wavs/00000.wav|he made mention that if your pastor|okyeame_speaker
```

**Stage 1 result:** 27,038 train / 3,005 eval / 493 skipped

---

### Step 4 — Fine-tune XTTS v2

Downloads the base XTTS v2 checkpoint and fine-tunes the GPT-2 decoder on the formatted dataset.

```bash
python training/train.py \
  --train_csv /kaggle/working/coqui_format/metadata_train.csv \
  --eval_csv /kaggle/working/coqui_format/metadata_eval.csv \
  --output_dir /kaggle/working \
  --epochs 1
```

**To resume from a saved checkpoint:**

```bash
python training/train.py \
  --train_csv /kaggle/working/coqui_format/metadata_train.csv \
  --eval_csv /kaggle/working/coqui_format/metadata_eval.csv \
  --output_dir /kaggle/working \
  --restore_path /kaggle/input/datasets/derricknyarko/okyeame-xtts-final-stage1/best_model_9013.pth \
  --epochs 1
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 5e-6 |
| Batch size | 3 |
| Mixed precision | False (float32) |
| Save frequency | every 9,000 steps |
| Checkpoint saving | best model only |

**Stage 1 training history:**

| Epoch | Train Loss (start) | Eval Loss | Best Model |
|---|---|---|---|
| 0 | 4.463 | 3.527 | best_model_9013.pth ✅ |
| 1 (partial) | ~3.3 | — | interrupted |
| 2 | 2.848 | 3.580 | no improvement |
| 3 | 2.404 | 3.692 | no improvement |

Eval loss plateaued at **3.527** after epoch 0. The model learned the Ghanaian accent. Further improvement requires more data (Stage 2).

> **Disk management (Kaggle):** Each model checkpoint is ~5.3GB. The trainer always saves two files after evaluation. Delete the unnumbered `best_model.pth` immediately after each epoch to avoid filling the 20GB `/kaggle/working` limit.

---

## Evaluation

### UTMOS — Reference Speaker Scoring

UTMOS (Unified TTS MOS) is an automated Mean Opinion Score predictor trained on human listening ratings. It scores audio quality on a 1–5 scale, mimicking how a human listener would rate naturalness. Used here to rank all 30,535 training chunks and identify the best reference speakers for voice cloning.

```python
import torch
import torchaudio

predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

def score_utmos(wav_path):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return predictor(wav, sr).item()
```

Run the full scoring pipeline:

```bash
python evaluation/evaluate.py \
  --model_path /kaggle/working/inference_model/best_model_9013.pth \
  --config_path /kaggle/working/inference_model/config.json \
  --ref_wavs_csv /kaggle/input/datasets/derricknyarko/okyeame-xtts-utmos-scores/okyeame-xtts-utmos-scores.csv \
  --output_dir /kaggle/working/eval_output \
  --top_n 20
```

UTMOS scores range 1–5. Prefer reference speakers scoring above 3.5. The top-20 scoring files from the corpus make excellent preset voices for a demo.

**Saved scores:** `derricknyarko/okyeame-xtts-utmos-scores` on Kaggle (~2 hours to score all 30,535 chunks on T4 GPU).

### Planned — Stage 2

- **WER (Word Error Rate):** Transcribe generated audio with Whisper and compare to input text. Measures intelligibility.
- **MOS (Mean Opinion Score):** Human listeners rate naturalness 1–5. Gold standard before HuggingFace release.
- **SECS (Speaker Encoder Cosine Similarity):** Cosine similarity between speaker embeddings of generated and reference audio. Measures voice cloning fidelity.

---

## Kaggle Workflow

Kaggle resets `/kaggle/working` between sessions. Follow this workflow each session:

```python
# Cell 1 — Install
!pip install datasets numpy soundfile torchaudio ctc-forced-aligner pyyaml scipy -q
!pip install git+https://github.com/idiap/coqui-ai-TTS.git -q

# Cell 2 — Clone repo
!git clone https://github.com/derricknyarko/okyeame-tts.git

# Cell 3 — Monkey patch (required before any TTS import)
import transformers.pytorch_utils as pt_utils
import torch

def isin_mps_friendly(elements, test_elements):
    return torch.isin(elements, test_elements)

pt_utils.isin_mps_friendly = isin_mps_friendly

# Cell 4 — Regenerate coqui_format wavs from saved aligned chunks
!cd /kaggle/working/okyeame-tts && python data/processing/03_format.py \
  --input_dir /kaggle/input/datasets/derricknyarko/okyeame-aligned-chunks-v2 \
  --audio_dir /kaggle/input/datasets/derricknyarko/okyeame-aligned-chunks-v2/aligned_audio \
  --output_dir /kaggle/working/coqui_format

# Cell 5 — Resume training from last saved checkpoint
!cd /kaggle/working/okyeame-tts && python training/train.py \
  --train_csv /kaggle/working/coqui_format/metadata_train.csv \
  --eval_csv /kaggle/working/coqui_format/metadata_eval.csv \
  --output_dir /kaggle/working \
  --restore_path /kaggle/input/datasets/derricknyarko/okyeame-xtts-final-stage1/best_model_9013.pth \
  --epochs 1
```

**After each epoch:** Save best model to a new Kaggle Dataset version. Delete the duplicate `best_model.pth` before saving to avoid uploading 10GB.

---

## Saved Kaggle Datasets

| Dataset | Contents | Notes |
|---|---|---|
| `derricknyarko/okyeame-clean-samples` | metadata.pkl + audio wavs | Step 1 output |
| `derricknyarko/okyeame-aligned-chunks-v2` | aligned_metadata.pkl + aligned_audio/ | Step 2 output |
| `derricknyarko/okyeame-coqui-format` | metadata_train.csv + metadata_eval.csv | Step 3 CSVs — regenerate wavs each session |
| `derricknyarko/okyeame-xtts-final-stage1` | best_model_9013.pth + config.json | Stage 1 final model |
| `derricknyarko/okyeame-xtts-utmos-scores` | okyeame-xtts-utmos-scores.csv | UTMOS scores for all 30,535 aligned chunks |

---

## Known Issues & Workarounds

### `isin_mps_friendly` ImportError
**Cause:** Removed from `transformers.pytorch_utils` in newer versions. coqui-ai-TTS still imports it.  
**Fix:** Apply the monkey patch before any TTS import (see Quick Start).

### Python 3.12 Compatibility
**Cause:** The original `TTS` PyPI package only supports Python ≤ 3.11.  
**Fix:** Use `git+https://github.com/idiap/coqui-ai-TTS.git`.

### `vocab.json not found` on inference
**Cause:** XTTS v2 requires `vocab.json` in the same directory as the checkpoint and config.  
**Fix:** Download from `https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json` and place alongside the model files.

### `[Errno 9] Bad file descriptor` after collection
**Cause:** HuggingFace streaming finalizes after the loop breaks at the target count.  
**Fix:** None needed — data is already saved. Verify with `len(metadata)`.

### Disk space on Kaggle (20GB limit)
**Cause:** Each model checkpoint is ~5.3GB. Trainer saves two files after each evaluation.  
**Fix:** Delete `best_model.pth` immediately after each epoch. Keep only `best_model_XXXX.pth`.

### `grad_norm: 0.0` during training
**Cause:** A reporting quirk in the XTTS v2 GPT trainer.  
**Fix:** Not a real issue — loss curves confirm normal learning.

---

## Roadmap

**Stage 1 (Complete)** — 10,000 sample subset → fine-tune XTTS v2 → validate pipeline → UTMOS evaluation → live demo

**Stage 2** — Larger, more diverse subset of the 2,700-hour dataset → retrain → WER + MOS evaluation → HuggingFace model release → open inference API

**Stage 3** — Full dataset training → conversational domain expansion → research paper → multilingual extension (Ghanaian English + Twi)

---

## Requirements

```
datasets
numpy
soundfile
torch
torchaudio
ctc-forced-aligner
pyyaml
scipy
git+https://github.com/idiap/coqui-ai-TTS.git
```

---

## Demo

A live demo is available on HuggingFace Spaces — paste any text, pick a preset Ghanaian voice or upload your own 6–10 second clip, and generate speech instantly.

**[▶ Try okyeame-tts on HuggingFace Spaces](https://huggingface.co/spaces/nyarderr/okyeame-tts)**

---

## Known Limitations

These limitations are inherited from the training dataset (`ghananlpcommunity/ghana-english-asr-2700hrs`):

- **Broadcast domain only:** The dataset consists of broadcast news speech. The model may not generalise as naturally to conversational Ghanaian English — informal speech, code-switching, and everyday register are underrepresented.
- **Speaker diversity:** Speaker diversity across the corpus has not been formally audited. The model may reflect the accent and prosody patterns of a narrower speaker demographic than the full range of Ghanaian English speakers.
- **Transcription quality:** Transcriptions may contain occasional errors in proper nouns, which can affect pronunciation of names and places.

Stage 2 and Stage 3 will address domain coverage through broader data selection and potential supplementary data collection.

---



Contributions are welcome. The project is at Stage 1 — there is meaningful work to be done at every level.

**Data**
The training corpus (`ghananlpcommunity/ghana-english-asr-2700hrs`) contains 2,700 hours of Ghanaian English speech — substantial in volume. The current limitation is domain: the dataset is broadcast news, which means the model may not generalise as well to conversational English. Stage 2 will train on a larger, more diverse subset. If you have access to conversational Ghanaian English recordings and can help expand the domain coverage, open an issue.

**Voices**
If you are a Ghanaian English speaker and would like your voice included as a preset in the demo, record a clean 10–30 second audio clip and open an issue with the recording attached.

**Code**
- Bug fixes and workaround improvements are always welcome
- Evaluation scripts (WER, MOS, SECS) are planned for Stage 2 — contributions welcome
- If you have experience with XTTS v2 training at scale, open an issue to discuss Stage 2 approach

**Research**
If you are a researcher working on African speech synthesis or low-resource TTS and want to collaborate on Stage 3 (full dataset, paper), reach out via GitHub issues.

Please open an issue before starting significant work so we can coordinate.

---



MIT

---

## Acknowledgements

- [Ghana NLP Community](https://huggingface.co/ghananlpcommunity) for the Ghana English ASR dataset (Owusu, Mich-Seth, 2026)
- [Idiap Research Institute](https://github.com/idiap/coqui-ai-TTS) for maintaining the Coqui TTS fork
- [Coqui TTS](https://github.com/coqui-ai/TTS) for the original XTTS v2 implementation
- [Intron Health](https://huggingface.co/intronhealth/afro-tts) for Afro-TTS — pan-African accented English TTS covering 86 African accents
- [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS) for UTMOS automated MOS prediction

## Citation

This project builds on the Ghana English ASR dataset:

```bibtex
@dataset{ghana_english_asr,
  author    = {Owusu, Mich-Seth},
  title     = {Ghana English ASR Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/ghananlpcommunity/ghana-english-asr-2700hrs}
}
```