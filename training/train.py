# training/train.py

import argparse
import gc
import os

import yaml
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsAudioConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
)
from TTS.utils.manage import ModelManager


def train(
    train_csv,
    eval_csv,
    output_dir,
    language="en",
    num_epochs=10,
    batch_size=3,
    grad_accum=1,
):
    RUN_NAME = "okyeame_xtts_ft"
    PROJECT_NAME = "okyeame-tts"
    OUT_PATH = os.path.join(output_dir, "run", "training")
    CHECKPOINTS_PATH = os.path.join(OUT_PATH, "XTTS_v2_original")
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

    # --- download base model files ---
    DVAE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
    MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"
    TOKENIZER_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
    CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"

    DVAE = os.path.join(CHECKPOINTS_PATH, "dvae.pth")
    MEL_NORM = os.path.join(CHECKPOINTS_PATH, "mel_stats.pth")
    TOKENIZER = os.path.join(CHECKPOINTS_PATH, "vocab.json")
    CHECKPOINT = os.path.join(CHECKPOINTS_PATH, "model.pth")
    

    if not os.path.isfile(DVAE) or not os.path.isfile(MEL_NORM):
        print("Downloading DVAE files...")
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_LINK], CHECKPOINTS_PATH, progress_bar=True
        )

    if not os.path.isfile(TOKENIZER) or not os.path.isfile(CHECKPOINT):
        print("Downloading XTTS v2 checkpoint...")
        ModelManager._download_model_files(
            [TOKENIZER_LINK, CHECKPOINT_LINK],
            CHECKPOINTS_PATH,
            progress_bar=True,
        )

    # --- dataset config ---
    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="okyeame_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=train_csv,
        meta_file_val=eval_csv,
        language=language,
    )

    # --- model args ---
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        max_wav_length=255995,  # ~11.6 secs
        max_text_length=200,
        mel_norm_file=MEL_NORM,
        dvae_checkpoint=DVAE,
        xtts_checkpoint=CHECKPOINT,
        tokenizer_file=TOKENIZER,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # --- audio config ---
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000
    )

    # --- trainer config ---
    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        audio=audio_config,
        batch_size=batch_size,
        batch_group_size=48,
        eval_batch_size=batch_size,
        num_loader_workers=4,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[],
        dashboard_logger="tensorboard",
        logger_uri=None,
        run_description="Okyeame TTS - Ghanaian English XTTS v2 fine-tuning",
    )

    # --- load samples ---
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")

    # --- init model ---
    model = GPTTrainer.init_from_config(config)

    # --- train ---
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=grad_accum,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()

    # cleanup
    del model, trainer, train_samples, eval_samples
    gc.collect()

    print(f"\nTraining complete! Model saved to {OUT_PATH}")


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="Fine-tune XTTS v2 on Ghanaian English"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=os.path.join(config["paths"]["coqui_output"], "metadata_train.csv"),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=os.path.join(config["paths"]["coqui_output"], "metadata_eval.csv"),
        help="Path to eval CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=config["paths"]["output"],
        help="Directory to save training output",
    )
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps"
    )
    args = parser.parse_args()

    train(
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        output_dir=args.output_dir,
        language=args.language,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
