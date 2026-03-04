# data/processing/03_format.py

import os
import csv
import pickle
import shutil


FILLERS = {"uh", "um", "hmm", "hm", "ah"}

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

def format_ljspeech(aligned_metadata_path, output_dir):
    """
    Converts aligned chunks into LJSpeech format for XTTS v2 fine-tuning.

    LJSpeech structure:
        dataset/
        ├── wavs/
        │   ├── 00000.wav
        │   └── ...
        └── metadata.csv  (filename|text|text)
    """
    os.makedirs(f"{output_dir}/wavs", exist_ok=True)

    with open(aligned_metadata_path, "rb") as f:
        saved = pickle.load(f)
        metadata = saved["metadata"]

    print(f"Formatting {len(metadata)} chunks into LJSpeech format...")

    csv_rows = []
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

        # copy wav to wavs folder
        filename = f"{i:05d}"
        dst = f"{output_dir}/wavs/{filename}.wav"
        shutil.copy(chunk["audio_path"], dst)

        # LJSpeech format: filename|text|text
        csv_rows.append([filename, chunk["text"], chunk["text"]])

    # write metadata.csv
    with open(f"{output_dir}/metadata.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerows(csv_rows)

    print(f"Done!")
    print(f"Total chunks formatted: {len(csv_rows)}")
    print(f"Skipped (no text or invalid): {skipped}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    format_ljspeech(
        aligned_metadata_path=config["paths"]["aligned_metadata"],
        output_dir=config["paths"]["ljspeech_output"],
    )
