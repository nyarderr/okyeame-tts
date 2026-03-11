"""
evaluate.py - Evaluate okyeame-tts model quality
Metrics:
  - UTMOS: automated MOS score (audio quality)
  - WER: Word Error Rate (intelligibility)
  - Speaker Similarity: voice cloning quality (optional, Stage 2)
"""

import argparse
import os

import pandas as pd
import torch
import torchaudio

# monkey patch
import transformers.pytorch_utils as pt_utils
import whisper
from jiwer import wer
from tqdm import tqdm
from TTS.api import TTS


def isin_mps_friendly(elements, test_elements):
    return torch.isin(elements, test_elements)


pt_utils.isin_mps_friendly = isin_mps_friendly


def score_utmos(wav_path, predictor):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return predictor(wav, sr).item()


def score_wer(text, wav_path, asr_model):
    result = asr_model.transcribe(wav_path)
    return wer(text.lower(), result["text"].lower())


def evaluate(model_path, config_path, vocab_path, ref_wavs, test_texts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load TTS
    tts = TTS()
    tts.load_tts_model_by_path(model_path=model_path, config_path=config_path)

    # Load UTMOS
    utmos = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )

    # Load Whisper for WER
    asr = whisper.load_model("base")

    results = []
    for text in tqdm(test_texts, desc="texts"):
        for ref_wav in tqdm(ref_wavs, desc="speakers", leave=False):
            speaker_id = os.path.basename(ref_wav)
            out_path = os.path.join(output_dir, f"gen_{speaker_id}")

            # Generate
            tts.tts_to_file(
                text=text, speaker_wav=ref_wav, language="en", file_path=out_path
            )

            # Score
            utmos_score = score_utmos(out_path, utmos)
            wer_score = score_wer(text, out_path, asr)

            results.append(
                {
                    "speaker": speaker_id,
                    "utmos": utmos_score,
                    "wer": wer_score,
                    "text": text[:50],
                    "output": out_path,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "eval_results.csv"), index=False)

    print("\n=== EVALUATION SUMMARY ===")
    print(f"Avg UTMOS: {df['utmos'].mean():.3f}")
    print(f"Avg WER:   {df['wer'].mean():.2%}")
    print(f"Best speaker by UTMOS: {df.loc[df['utmos'].idxmax(), 'speaker']}")
    print(f"Best speaker by WER:   {df.loc[df['wer'].idxmin(), 'speaker']}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--ref_wavs_dir", required=True)
    parser.add_argument("--output_dir", default="/kaggle/working/eval_output")
    parser.add_argument("--top_n_speakers", type=int, default=20)
    args = parser.parse_args()

    # Load top N speakers from UTMOS scores if available
    utmos_csv = os.path.join(args.ref_wavs_dir, "utmos_scores.csv")
    if os.path.exists(utmos_csv):
        df_scores = pd.read_csv(utmos_csv).sort_values("utmos", ascending=False)
        ref_wavs = df_scores["path"].tolist()[: args.top_n_speakers]
    else:
        ref_wavs = [
            os.path.join(args.ref_wavs_dir, f)
            for f in os.listdir(args.ref_wavs_dir)[: args.top_n_speakers]
        ]

    test_texts = [
        "The government has announced new policies to support local businesses in Ghana.",
        "Please call your mother, she has been trying to reach you since morning.",
        "In Neo-Colonialism, the last stage of Imperialism, Kwame Nkrumah warned that a state can have nominal independence while its economic policy is dictated externally.",
    ]

    evaluate(
        model_path=args.model_path,
        config_path=args.config_path,
        vocab_path=args.vocab_path,
        ref_wavs=ref_wavs,
        test_texts=test_texts,
        output_dir=args.output_dir,
    )
