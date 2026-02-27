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

def collate_fn(batch):
    # batch: list of (src_tensor, tgt_tensor)
    srcs, tgts = zip(*batch)
    PAD = None
    # all examples share same sp model and PAD; get from first
    PAD = max([ (x.max().item() if x.numel()>0 else 0) for x in srcs + tgts ])  # fallback
    # safer: expect user to provide PAD externally; for simplicity, find pad as the largest id+1 - not ideal but works
    src_lens = [s.size(0) for s in srcs]
    tgt_lens = [t.size(0) for t in tgts]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    # choose pad id from dataset knowledge: assume PAD = vocab_size+2, so use max id in batch +1 if that is larger
    pad = max([int(s.max().item()) for s in srcs + tgts]) + 1

    padded_src = torch.full((len(batch), max_src), pad, dtype=torch.long)
    padded_tgt = torch.full((len(batch), max_tgt), pad, dtype=torch.long)
    for i,(s,t) in enumerate(batch):
        padded_src[i, :s.size(0)] = s
        padded_tgt[i, :t.size(0)] = t

    src_lens = torch.LongTensor(src_lens)
    tgt_lens = torch.LongTensor(tgt_lens)
    return padded_src, src_lens, padded_tgt, tgt_lens