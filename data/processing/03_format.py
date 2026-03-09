# data/processing/03_format.py

import csv
import os
import pickle

import numpy as np
import soundfile as sf
from scipy import signal

FILLERS = {"uh", "um", "hmm", "hm", "ah"}
SPEAKER_NAME = "okyeame_speaker"
TARGET_SR = 22050


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    # calculate resampling ratio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)
    resampled = signal.resample(audio, new_length)
    return resampled.astype(np.float32)


def is_valid_chunk(chunk):
    text = chunk["text"].strip()
    words = text.split()

    # skip if too long
    if len(text) > 150:
        return False

    # skip if more than 1 filler word
    filler_count = sum(1 for w in words if w.lower() in FILLERS)
    if filler_count > 1:
        return False

    return True


def format_coqui(aligned_metadata_path, output_dir, eval_split=0.1):
    """
    Converts aligned chunks into Coqui format for XTTS v2 fine-tuning.

    LJSpeech structure:
        dataset/
        ├── wavs/
        │   ├── 00000.wav (resampled to 22050 Hz)
        │   └── ...
        └── metadata_train.csv
        └── metadata_eval.csv
    """
    os.makedirs(f"{output_dir}/wavs", exist_ok=True)

    with open(aligned_metadata_path, "rb") as f:
        saved = pickle.load(f)
        metadata = saved["metadata"]

    print(f"Formatting {len(metadata)} chunks")

    rows = []
    skipped = 0

    for i, chunk in enumerate(metadata):
        # skip chunks with missing text
        if not chunk["text"].strip():
            skipped += 1
            continue

        # skip invalid chunks
        if not is_valid_chunk(chunk):
            skipped += 1
            continue

        # read and resample audio
        audio, sr = sf.read(chunk["audio_path"])
        audio = audio.astype(np.float32)
        audio = resample_audio(audio, sr, TARGET_SR)

        # save resampled wav
        filename = f"{i:05d}"
        wav_path = f"{output_dir}/wavs/{filename}.wav"
        sf.write(wav_path, audio, TARGET_SR)

        # coqui format: audio_file|text|speaker_name
        rows.append([f"wavs/{filename}.wav", chunk["text"], SPEAKER_NAME])

    # split into train and eval
    split_idx = int(len(rows) * (1 - eval_split))
    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:]

    # save metadata csvs
    for split, split_rows in zip(["train", "eval"], [train_rows, eval_rows]):
        with open(f"{output_dir}/metadata_{split}.csv", "w", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(split_rows)

    print("Done!")
    print(f"Train samples: {len(train_rows)}")
    print(f"Eval samples: {len(eval_rows)}")
    print(f"Skipped: {skipped}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    import argparse

    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Format aligned chunks into Coqui format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=config["paths"]["output"],
        help="Path to aligned metadata pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config["paths"]["coqui_output"],
        help="Directory to save Coqui format output",
    )
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.1,
        help="Proportion of data to use for evaluation",
    )

    args = parser.parse_args()

    # derive aligned_metadata path from input_dir
    aligned_metadata_path = os.path.join(args.input_dir, "aligned_metadata.pkl")

    format_coqui(
        aligned_metadata_path=aligned_metadata_path,
        output_dir=args.output_dir,
        eval_split=args.eval_split,
    )
