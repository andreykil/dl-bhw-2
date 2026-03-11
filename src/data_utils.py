# src/data_utils.py

import random
import torch
from torch.utils.data import Sampler, DataLoader
import sacrebleu
from pathlib import Path
from typing import List, Optional, Dict

from src.inference import translate_file

class TokenBatchSampler(Sampler):

    def __init__(
        self,
        dataset,
        tokens_per_batch=8000,
        pool_size=10000,
        shuffle=True,
    ):
        self.dataset = dataset
        self.tokens_per_batch = tokens_per_batch
        self.pool_size = pool_size
        self.shuffle = shuffle

        self.src_lens = [len(dataset.encode(x)) for x in dataset.src_lines]
        self.tgt_lens = [len(dataset.encode(x)) for x in dataset.tgt_lines]

    def __iter__(self):

        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        batches = []

        for i in range(0, len(indices), self.pool_size):

            pool = indices[i:i+self.pool_size]

            # sort only inside pool
            pool.sort(key=lambda x: max(self.src_lens[x], self.tgt_lens[x]))

            batch = []
            max_src = 0
            max_tgt = 0

            for idx in pool:

                src_len = self.src_lens[idx]
                tgt_len = self.tgt_lens[idx]

                new_src = max(max_src, src_len)
                new_tgt = max(max_tgt, tgt_len)

                new_size = len(batch) + 1
                tokens = (new_src + new_tgt) * new_size

                if tokens <= self.tokens_per_batch or len(batch) == 0:

                    batch.append(idx)
                    max_src = new_src
                    max_tgt = new_tgt

                else:

                    batches.append(batch)
                    batch = [idx]
                    max_src = src_len
                    max_tgt = tgt_len

            if batch:
                batches.append(batch)

        # shuffle batches
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

class CollateFn:
    """Callable class для collate"""
    
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        srcs, tgts = zip(*batch)
        
        max_src = max(len(x) for x in srcs)
        max_tgt = max(len(x) for x in tgts)
        
        src_tensor = torch.full((len(batch), max_src), self.pad_idx)
        tgt_tensor = torch.full((len(batch), max_tgt), self.pad_idx)
        
        for i, (s, t) in enumerate(batch):
            src_tensor[i, :len(s)] = s
            tgt_tensor[i, :len(t)] = t
        
        src_lens = torch.LongTensor([len(x) for x in srcs])
        tgt_lens = torch.LongTensor([len(x) for x in tgts])
        
        return src_tensor, src_lens, tgt_tensor, tgt_lens


def create_token_dataloader(dataset, tokens_per_batch, shuffle=True):

    sampler = TokenBatchSampler(
        dataset,
        tokens_per_batch=tokens_per_batch,
        shuffle=shuffle
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=CollateFn(dataset.PAD),
        num_workers=2
    )


def evaluate_model_on_validation(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 64,
    max_decoding_len: int = 50,
    beam_size: int = 5,
    no_repeat_ngram_size: int = 3,
    run_greedy: bool = True,
    run_beam: bool = True,
    n_examples: int = 10
) -> Dict[str, Optional[object]]:

    model.eval()
    model.to(device)

    src_lines: List[str] = list(dataset.src_lines)
    ref_lines: List[str] = list(dataset.tgt_lines)

    results: Dict[str, Optional[object]] = {
        "preds_greedy": None,
        "preds_beam": None,
        "bleu_greedy": None,
        "bleu_beam": None
    }

    # ---- GREEDY ----
    if run_greedy:

        preds_greedy: List[str] = translate_file(
            model=model,
            dataset=dataset,
            input_lines=src_lines,
            max_decoding_len=max_decoding_len,
            device=device,
            batch_size=batch_size,
            mode="greedy",
        )

        bleu_g = sacrebleu.corpus_bleu(preds_greedy, [ref_lines]).score

        results["preds_greedy"] = preds_greedy
        results["bleu_greedy"] = float(bleu_g)

        print(f"GREEDY BLEU: {bleu_g:.2f}")

    # ---- BEAM ----
    if run_beam:

        preds_beam: List[str] = translate_file(
            model=model,
            dataset=dataset,
            input_lines=src_lines,
            max_decoding_len=max_decoding_len,
            device=device,
            batch_size=batch_size,
            mode="beam",
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        bleu_b = sacrebleu.corpus_bleu(preds_beam, [ref_lines]).score

        results["preds_beam"] = preds_beam
        results["bleu_beam"] = float(bleu_b)

        print(f"BEAM BLEU: {bleu_b:.2f}")

    # ---- ПРИМЕРЫ ----
    if run_beam and results["preds_beam"] is not None:
        preds = results["preds_beam"]
    else:
        preds = results["preds_greedy"]

    print(f"\ncounts: pred={len(preds)}, ref={len(ref_lines)}") # type: ignore

    n_show = min(n_examples, len(preds)) # type: ignore

    for i in range(n_show):

        print(f"\n=== Example {i} ===")

        if run_greedy and results["preds_greedy"] is not None:
            print("PRED greedy:", repr(results["preds_greedy"][i])) # type: ignore

        if run_beam and results["preds_beam"] is not None:
            print("PRED beam:  ", repr(results["preds_beam"][i])) # type: ignore

        print("REF:        ", repr(ref_lines[i]))

    return results