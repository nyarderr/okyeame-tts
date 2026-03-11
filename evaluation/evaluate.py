"""
evaluate.py - Evaluate okyeame-tts model quality using UTMOS
"""

import os
import argparse
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from TTS.api import TTS

import transformers.pytorch_utils as pt_utils


def isin_mps_friendly(elements, test_elements):
    return torch.isin(elements, test_elements)


pt_utils.isin_mps_friendly = isin_mps_friendly


def score_utmos(wav_path, predictor):
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return predictor(wav, sr).item()


def evaluate(model_path, config_path, ref_wavs, test_texts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    tts = TTS()
    tts.load_tts_model_by_path(model_path=model_path, config_path=config_path)

    utmos = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )

    results = []
    for text in tqdm(test_texts, desc="texts"):
        for ref_wav in tqdm(ref_wavs, desc="speakers", leave=False):
            speaker_id = os.path.basename(ref_wav)
            out_path = os.path.join(output_dir, f"gen_{speaker_id}")

            tts.tts_to_file(
                text=text, speaker_wav=ref_wav, language="en", file_path=out_path
            )

            results.append(
                {
                    "speaker": speaker_id,
                    "utmos": score_utmos(out_path, utmos),
                    "text": text[:60],
                    "output": out_path,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "eval_results.csv"), index=False)

    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Avg UTMOS: {df['utmos'].mean():.3f}")
    print(
        f"Best speaker: {df.loc[df['utmos'].idxmax(), 'speaker']} ({df['utmos'].max():.3f})"
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ref_wavs_csv", required=True, help="CSV with utmos scores")
    parser.add_argument("--output_dir", default="/kaggle/working/eval_output")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()

    df_scores = pd.read_csv(args.ref_wavs_csv).sort_values("utmos", ascending=False)
    ref_wavs = df_scores["path"].tolist()[: args.top_n]

    test_texts = [
        "The government has announced new policies to support local businesses in Ghana.",
        "In Neo-Colonialism, the last stage of Imperialism, Kwame Nkrumah warned that a state can have nominal independence.",
    ]

    evaluate(args.model_path, args.config_path, ref_wavs, test_texts, args.output_dir)
