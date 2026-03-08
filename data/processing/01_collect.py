import os
import pickle
import time

import numpy as np
import soundfile as sf
from datasets import load_dataset


def compute_snr(audio_array):
    signal = np.mean(audio_array**2)
    noise = np.mean(np.abs(np.diff(audio_array)) ** 2)
    if noise < 1e-9 or signal < 1e-9:
        return 0
    return 10 * np.log10(signal / noise)


def is_clean(sample):
    audio = np.array(sample["audio"]["array"])
    duration = sample["duration_ss"]
    text = sample["corrected_text"]
    snr = compute_snr(audio)
    return snr > 2 and 2.0 <= duration <= 16.0 and 20 < len(text.strip()) < 300


def save_progress(metadata, total_seen, skipped, output_dir):
    with open(f"{output_dir}/metadata.pkl", "wb") as f:
        pickle.dump(
            {"metadata": metadata, "total_seen": total_seen, "skipped": skipped}, f
        )
    print(f"Saved {len(metadata)} samples")


def collect(dataset_name, output_dir, target=10000):
    os.makedirs(f"{output_dir}/audio", exist_ok=True)

    # resume if checkpoint exists
    try:
        with open(f"{output_dir}/metadata.pkl", "rb") as f:
            saved = pickle.load(f)
            metadata = saved["metadata"]
            total_seen = saved["total_seen"]
            skipped = saved["skipped"]
        print(f"Resuming from {len(metadata)} samples")
    except:
        metadata = []
        total_seen = 0
        skipped = 0
        print("Starting fresh")

    while len(metadata) < target:
        try:
            print("Connecting to dataset...")
            dataset = load_dataset(dataset_name, split="train", streaming=True)

            for sample in dataset:
                total_seen += 1
                try:
                    if is_clean(sample):
                        audio = np.array(sample["audio"]["array"], dtype=np.float32)
                        sr = sample["audio"]["sampling_rate"]
                        idx = len(metadata)
                        audio_path = f"{output_dir}/audio/{idx:05d}.wav"
                        sf.write(audio_path, audio, sr)

                        metadata.append(
                            {
                                "text": sample["corrected_text"],
                                "duration": sample["duration_ss"],
                                "sampling_rate": sr,
                                "audio_path": audio_path,
                            }
                        )
                except:
                    skipped += 1
                    continue

                if total_seen % 50 == 0:
                    print(
                        f"Seen: {total_seen} | Clean: {len(metadata)} | Skipped: {skipped}"
                    )

                if len(metadata) % 500 == 0 and len(metadata) > 0:
                    save_progress(metadata, total_seen, skipped, output_dir)

                if len(metadata) >= target:
                    break

        except Exception as e:
            print(f"Connection dropped: {e}")
            save_progress(metadata, total_seen, skipped, output_dir)
            time.sleep(15)
            continue

        break

    save_progress(metadata, total_seen, skipped, output_dir)
    print(f"Done! Collected: {len(metadata)} | Skipped: {skipped}")
    return metadata


if __name__ == "__main__":
    
    import yaml
    import os

    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    
    collect(
        dataset_name= os.environ.get('DATASET_NAME', config['dataset']['name']),
        output_dir= os.environ.get('OUTPUT_DIR', config['paths']['output']),
        target= os.environ.get('TARGET', config['dataset']['target']),
    )
