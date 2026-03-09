# okyeame-tts

> **Okyeame** (ɔkyeame) — the royal linguist and spokesperson in Ghanaian tradition, renowned for clarity, eloquence, and the power of voice.

A fine-tuned XTTS v2 text-to-speech model that speaks English with a natural Ghanaian accent. Built on the `ghananlpcommunity/ghana-english-asr-2700hrs` dataset, okyeame-tts is the first open-source TTS model trained specifically on Ghanaian English speech.

---

## Why This Exists

Commercial TTS systems like ElevenLabs support voice cloning — but they require a reference audio clip every time. okyeame-tts is different: it speaks Ghanaian English **by default**, with no reference audio needed. Ghanaian English is the native voice, not an imitation.

---

## Project Status

| Stage | Status | Description |
|---|---|---|
| Data Collection | ✅ Done | 10,000 samples, 37hrs from ASR dataset |
| Forced Alignment | 🔄 In Progress | Chunking audio with text alignment |
| LJSpeech Formatting | ⏳ Pending | Coqui format with train/eval split |
| XTTS v2 Fine-tuning | ⏳ Pending | Fine-tune on aligned Ghanaian English |
| Evaluation | ⏳ Pending | MOS scoring + accent evaluation |
| HuggingFace Release | ⏳ Pending | Stage 2 (50k-100k samples) |

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
│   └── evaluate.py                # MOS scoring and accent evaluation
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

Due to a version incompatibility between `coqui-ai-TTS` and newer versions of `transformers`, a missing function must be patched before importing TTS modules. Run this **once per environment** before running any script:

```python
import transformers.pytorch_utils as pt_utils
import torch

def isin_mps_friendly(elements, test_elements):
    return torch.isin(elements, test_elements)

pt_utils.isin_mps_friendly = isin_mps_friendly
```

Or patch the transformers file directly (persists across sessions):

```python
patch = '''
import torch

def isin_mps_friendly(elements, test_elements):
    return torch.isin(elements, test_elements)
'''

file_path = '/path/to/site-packages/transformers/pytorch_utils.py'

with open(file_path, 'r') as f:
    content = f.read()

if 'isin_mps_friendly' not in content:
    with open(file_path, 'a') as f:
        f.write(patch)
    print("Patched!")
else:
    print("Already patched!")
```

> **Root cause:** `isin_mps_friendly` was removed from `transformers.pytorch_utils` in newer versions of the `transformers` library. The coqui-ai-TTS fork still imports it from there. The fix adds the function back using `torch.isin`, which is the underlying implementation.

---

## Configuration

All paths and dataset parameters are set in `config.yaml` at the project root. Scripts read from this file and allow any value to be overridden via command-line arguments.

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

Update `config.yaml` for your environment or override any value at runtime using `--argument` flags (see each script below).

---

## Pipeline

The full pipeline runs in four steps. Each step produces output consumed by the next.

```
01_collect.py  →  metadata.pkl + audio/*.wav
      ↓
02_align.py    →  aligned_metadata.pkl + aligned_audio/*.wav
      ↓
03_format.py   →  wavs/*.wav + metadata_train.csv + metadata_eval.csv
      ↓
train.py       →  best_model.pth (fine-tuned XTTS v2)
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

**Arguments:**

| Argument | Default (from config) | Description |
|---|---|---|
| `--dataset_name` | `ghananlpcommunity/ghana-english-asr-2700hrs` | HuggingFace dataset name |
| `--output_dir` | `config.paths.output` | Directory to save metadata.pkl and audio/ |
| `--target` | `config.dataset.target` | Number of samples to collect |

**Output:**
```
output_dir/
├── metadata.pkl          # list of {text, audio_path, duration, sampling_rate}
└── audio/
    ├── 00000.wav
    ├── 00001.wav
    └── ...
```

---

### Step 2 — Forced Alignment

Uses `ctc-forced-aligner` to align transcripts to audio at the word level, then splits audio into natural chunks of 2–7 seconds at pauses and punctuation boundaries.

```bash
python data/processing/02_align.py \
  --metadata_path /kaggle/working/metadata.pkl \
  --audio_dir /kaggle/working/audio \
  --output_dir /kaggle/working \
  --target 10000
```

**Arguments:**

| Argument | Default (from config) | Description |
|---|---|---|
| `--metadata_path` | `config.paths.metadata` | Path to metadata.pkl from Step 1 |
| `--audio_dir` | `config.paths.audio_dir` | Directory containing audio wav files |
| `--output_dir` | `config.paths.output` | Directory to save aligned output |
| `--target` | `config.dataset.target` | Number of samples to process |

**Checkpointing:** The script saves a checkpoint every 100 samples to `output_dir/aligned_metadata.pkl`. If the session is interrupted, rerunning the script automatically resumes from the last checkpoint.

**Output:**
```
output_dir/
├── aligned_metadata.pkl     # list of {text, audio_path, duration, sampling_rate}
└── aligned_audio/
    ├── 00000.wav             # 2–7 second aligned chunks
    ├── 00001.wav
    └── ...
```

**Expected runtime:**
- GPU (P100): ~3.8s per sample → ~10 hours for 10,000 samples
- CPU: ~30s per sample → ~80 hours for 10,000 samples

> **Tip (Kaggle):** Run overnight on GPU. Monitor the ETA printed every 50 samples. Save `aligned_metadata.pkl` as a Kaggle Dataset immediately after completion — `/kaggle/working` is wiped when the session ends.

---

### Step 3 — Coqui Format

Converts aligned chunks into the Coqui CSV format expected by XTTS v2 fine-tuning. Resamples audio from 16000Hz to 22050Hz and splits into train/eval sets.

```bash
python data/processing/03_format.py \
  --input_dir /kaggle/working \
  --output_dir /kaggle/working/coqui_format \
  --eval_split 0.1
```

**Arguments:**

| Argument | Default (from config) | Description |
|---|---|---|
| `--input_dir` | `config.paths.output` | Directory containing aligned_metadata.pkl |
| `--output_dir` | `config.paths.ljspeech_output` | Directory to save formatted dataset |
| `--eval_split` | `0.1` | Fraction of data for evaluation |

**Filters applied:**
- Skips chunks with empty text
- Skips chunks with text longer than 150 characters
- Skips chunks with more than 1 filler word (uh, um, hmm, hm, ah)

**Output:**
```
output_dir/
├── wavs/
│   ├── 00000.wav          # resampled to 22050Hz
│   └── ...
├── metadata_train.csv     # 90% of data
└── metadata_eval.csv      # 10% of data
```

**CSV format (Coqui):**
```
audio_file|text|speaker_name
wavs/00000.wav|he made mention that if your pastor|okyeame_speaker
wavs/00001.wav|there is no two ways about it|okyeame_speaker
```

---

### Step 4 — Fine-tune XTTS v2

Downloads the base XTTS v2 checkpoint and fine-tunes it on your formatted Ghanaian English dataset using the GPT trainer.

```bash
python training/train.py \
  --train_csv /kaggle/working/coqui_format/metadata_train.csv \
  --eval_csv /kaggle/working/coqui_format/metadata_eval.csv \
  --output_dir /kaggle/working \
  --epochs 10 \
  --batch_size 3
```

**Arguments:**

| Argument | Default (from config) | Description |
|---|---|---|
| `--train_csv` | derived from `config.paths.ljspeech_output` | Path to training CSV |
| `--eval_csv` | derived from `config.paths.ljspeech_output` | Path to eval CSV |
| `--output_dir` | `config.paths.output` | Directory to save model checkpoints |
| `--language` | `en` | Language code |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `3` | Batch size per step |
| `--grad_accum` | `1` | Gradient accumulation steps |

**What gets downloaded automatically:**
- `dvae.pth` — Discrete VAE checkpoint
- `mel_stats.pth` — Mel spectrogram normalization stats
- `vocab.json` — XTTS v2 tokenizer
- `model.pth` — Base XTTS v2 checkpoint (1.87GB)

**Output:**
```
output_dir/run/training/
├── XTTS_v2_original/
│   ├── dvae.pth
│   ├── mel_stats.pth
│   ├── vocab.json
│   └── model.pth
└── okyeame_xtts_ft-{date}/
    ├── best_model.pth       # best checkpoint by eval loss
    ├── checkpoint_1000.pth  # periodic checkpoint
    └── config.json          # training configuration
```

**Expected runtime (GPU P100):**
- ~7 minutes per epoch
- 10 epochs ≈ ~70 minutes

---

## Running in Kaggle

Kaggle resets `/kaggle/working` between sessions. Follow this workflow:

```python
# Cell 1 — Install dependencies
!pip install datasets numpy soundfile torchaudio ctc-forced-aligner pyyaml scipy -q
!pip install git+https://github.com/idiap/coqui-ai-TTS.git -q

# Cell 2 — Clone repo
!git clone https://github.com/derricknyarko/okyeame-tts.git

# Cell 3 — Apply monkey patch (required)
import transformers.pytorch_utils as pt_utils
import torch

def isin_mps_friendly(elements, test_elements):
    return torch.isin(elements, test_elements)

pt_utils.isin_mps_friendly = isin_mps_friendly

# Cell 4 — Run pipeline
!cd /kaggle/working/okyeame-tts && python data/processing/01_collect.py --output_dir /kaggle/working --target 10000
!cd /kaggle/working/okyeame-tts && python data/processing/02_align.py --metadata_path /kaggle/working/metadata.pkl --audio_dir /kaggle/working/audio --output_dir /kaggle/working --target 10000
!cd /kaggle/working/okyeame-tts && python data/processing/03_format.py --input_dir /kaggle/working --output_dir /kaggle/working/coqui_format
!cd /kaggle/working/okyeame-tts && python training/train.py --train_csv /kaggle/working/coqui_format/metadata_train.csv --eval_csv /kaggle/working/coqui_format/metadata_eval.csv --output_dir /kaggle/working
```

> **Important:** Save intermediate outputs as Kaggle Datasets after each major step. `/kaggle/working` is temporary. Committing a notebook only saves the notebook, not the data files.

---

## Dataset

**Source:** `ghananlpcommunity/ghana-english-asr-2700hrs`  
**Size used:** 10,000 samples (~37 hours)  
**Language:** Ghanaian English  
**Sample rate:** 16000Hz (resampled to 22050Hz for training)

The dataset contains transcribed Ghanaian English speech collected from community members across Ghana. It covers a range of speakers, topics, and recording conditions.

---

## Known Issues & Workarounds

### `isin_mps_friendly` ImportError
**Cause:** `coqui-ai-TTS` imports `isin_mps_friendly` from `transformers.pytorch_utils`, which was removed in newer versions of `transformers`.  
**Fix:** See [Apply the Monkey Patch](#2-apply-the-monkey-patch-required) above.

### Python 3.12 Compatibility
**Cause:** The original `TTS` package on PyPI only supports up to Python 3.11.  
**Fix:** Use `git+https://github.com/idiap/coqui-ai-TTS.git` — the Idiap community fork with Python 3.12 support.

### `[Errno 9] Bad file descriptor` after collection
**Cause:** HuggingFace streaming connection tries to fetch more data after the target count is reached and the loop breaks. This is a cleanup issue, not a data error.  
**Fix:** None needed — collection completes successfully before this error appears. Verify with `len(metadata)`.

### `PyGILState_Release` fatal error
**Cause:** Same as above — Python runtime finalizing while HuggingFace streaming threads are still active.  
**Fix:** None needed — data is already saved before this occurs.

---

## Roadmap

**Stage 1 (Current)** — 10,000 samples → fine-tune XTTS v2 → validate pipeline → generate blog audio

**Stage 2** — 50,000–100,000 samples → retrain → publish on HuggingFace → open API

**Stage 3** — Full dataset → research paper → multilingual extension (Ghanaian English + Twi)

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

## License

MIT

---

## Acknowledgements

- [Ghana NLP Community](https://huggingface.co/ghananlpcommunity) for the Ghana English ASR dataset
- [Idiap Research Institute](https://github.com/idiap/coqui-ai-TTS) for maintaining the Coqui TTS fork
- [Coqui TTS](https://github.com/coqui-ai/TTS) for the original XTTS v2 implementation
