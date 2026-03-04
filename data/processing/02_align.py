import os
import pickle
import tempfile
import time

import numpy as np
import soundfile as sf
import yaml
from ctc_forced_aligner import get_word_stamps


def split_on_boundaries(word_stamps, audio, sr, min_dur=2.0, max_dur=7.0):
    chunks = []
    current_words = []
    current_start = None

    for i, word in enumerate(word_stamps):
        if current_start is None:
            current_start = word["start"]

        current_words.append(word)
        current_duration = word["end"] - current_start

        is_last_word = i == len(word_stamps) - 1

        if i < len(word_stamps) - 1:
            pause = word_stamps[i + 1]["start"] - word["end"]
        else:
            pause = 999

        is_natural_break = pause > 0.3 or word["text"].endswith((",", ".", "?"))

        if current_duration >= min_dur and (
            is_natural_break or current_duration >= max_dur or is_last_word
        ):
            start_sample = int(current_start * sr)
            end_sample = int(word["end"] * sr)
            audio_chunk = audio[start_sample:end_sample]

            chunks.append(
                {
                    "text": " ".join(w["text"] for w in current_words),
                    "audio": audio_chunk,
                    "sampling_rate": sr,
                    "duration": word["end"] - current_start,
                }
            )

            current_words = []
            current_start = None

    return chunks


def save_checkpoint(aligned_chunks, processed_idx, failed, output_dir):
    os.makedirs(f"{output_dir}/aligned_audio", exist_ok=True)

    # save audio files + metadata separately
    aligned_metadata = []
    for i, chunk in enumerate(aligned_chunks):
        audio_path = f"{output_dir}/aligned_audio/{i:05d}.wav"
        if not os.path.exists(audio_path):
            sf.write(audio_path, chunk["audio"], chunk["sampling_rate"])
        aligned_metadata.append(
            {
                "text": chunk["text"],
                "duration": chunk["duration"],
                "sampling_rate": chunk["sampling_rate"],
                "audio_path": audio_path,
            }
        )

    with open(f"{output_dir}/aligned_metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "metadata": aligned_metadata,
                "processed_idx": processed_idx,
                "failed": failed,
            },
            f,
        )

    print(f"Checkpoint — {len(aligned_metadata)} chunks from {processed_idx} samples")


def run_alignment(metadata, output_dir, target=10000, checkpoint_every=100):
    os.makedirs(f"{output_dir}/aligned_audio", exist_ok=True)

    # resume from checkpoint if exists
    checkpoint_path = f"{output_dir}/aligned_metadata.pkl"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            saved = pickle.load(f)
            aligned_chunks = saved["metadata"]
            start_idx = saved["processed_idx"]
            failed = saved["failed"]
        print(f"Resuming from sample {start_idx} with {len(aligned_chunks)} chunks")
    else:
        aligned_chunks = []
        start_idx = 0
        failed = []
        print("Starting fresh alignment")

    start_time = time.time()

    for idx in range(start_idx, min(target, len(metadata))):
        sample = metadata[idx]

        try:
            audio, sr = sf.read(sample["audio_path"])
            audio = audio.astype(np.float32)
            text = sample["text"]

            # write temp transcript file
            with tempfile.NamedTemporaryFile(
                suffix=".txt", delete=False, mode="w"
            ) as tf:
                tf.write(text)
                transcript_path = tf.name

            # run forced alignment
            result = get_word_stamps(
                audio_path=sample["audio_path"], transcript_path=transcript_path
            )
            word_stamps = result[0]

            # split into chunks
            chunks = split_on_boundaries(word_stamps, audio, sr)
            aligned_chunks.extend(chunks)

            os.unlink(transcript_path)

        except Exception as e:
            failed.append({"idx": idx, "error": str(e)})
            continue

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (idx - start_idx + 1) / elapsed
            remaining = (target - idx - 1) / rate
            print(
                f"Sample {idx + 1}/{target} | Chunks: {len(aligned_chunks)} | "
                f"Failed: {len(failed)} | ETA: {remaining / 3600:.1f}hrs"
            )

        if (idx + 1) % checkpoint_every == 0:
            save_checkpoint(aligned_chunks, idx + 1, failed, output_dir)

    save_checkpoint(aligned_chunks, target, failed, output_dir)

    print("\nAlignment complete!")
    print(f"Total chunks: {len(aligned_chunks)}")
    print(f"Failed: {len(failed)}")

    durations = [c["duration"] for c in aligned_chunks]
    print(f"Total hours: {sum(durations) / 3600:.2f}h")
    print(f"Avg duration: {np.mean(durations):.1f}s")

    return aligned_chunks


if __name__ == "__main__":
    import os
    import pickle

    import yaml

    # load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # load metadata
    with open(config["paths"]["metadata"], "rb") as f:
        saved = pickle.load(f)
        metadata = saved["metadata"]

    # remap audio paths to actual location
    audio_dir = config["paths"]["audio_dir"]
    for sample in metadata:
        filename = os.path.basename(sample["audio_path"])
        sample["audio_path"] = f"{audio_dir}/{filename}"

    run_alignment(metadata, output_dir=config["paths"]["output"])
