# src/dataset.py
import torch
from torch.utils.data import Dataset
import sentencepiece as spm
from pathlib import Path

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, sp_model, DATA_DIR='data'):
        self.src_lines = [l.rstrip("\n") for l in open(Path(DATA_DIR)/src_file, encoding="utf-8")]
        self.tgt_lines = [l.rstrip("\n") for l in open(Path(DATA_DIR)/tgt_file, encoding="utf-8")]
        assert len(self.src_lines) == len(self.tgt_lines)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(Path(DATA_DIR)/sp_model)) # type: ignore
        self.vocab_size = len(self.sp)  # base vocab size
        # reserve extra ids for BOS/EOS/PAD
        self.BOS = self.vocab_size
        self.EOS = self.vocab_size + 1
        self.PAD = self.vocab_size + 2

    def __len__(self):
        return len(self.src_lines)

    def encode(self, text):
        ids = self.sp.encode_as_ids(text) # type: ignore
        # add BOS and EOS
        return [self.BOS] + ids + [self.EOS]

    def __getitem__(self, idx):
        src = self.encode(self.src_lines[idx])
        tgt = self.encode(self.tgt_lines[idx])
        return torch.LongTensor(src), torch.LongTensor(tgt)
