# src/prepare_spm.py

from pathlib import Path
import sentencepiece as spm

def train_sentencepiece(
    data_dir="data",
    vocab_size=8000,
    model_prefix="sentencepiece",
    model_type="unigram",
    character_coverage=1.0
):

    data_dir = Path(data_dir)
    spm_input = data_dir / "spm_input.txt"

    # объединяем train.de + train.en
    with open(spm_input, "w", encoding="utf-8") as out:
        for name in ["train.de-en.de", "train.de-en.en"]:
            with open(data_dir / name, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.write(line + "\n")

    spm.SentencePieceTrainer.Train(
        input=str(spm_input),
        model_prefix=str(data_dir / model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage
    )

    print(f"SentencePiece model saved to {data_dir / (model_prefix + '.model')}")