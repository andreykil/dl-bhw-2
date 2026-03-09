# src/data_utils.py

import random
import torch
from torch.utils.data import Sampler, DataLoader

class TokenBatchSampler(Sampler):
    """
    Формирует батчи по количеству токенов.
    """

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

        # заранее считаем длины
        self.src_lens = [len(dataset.encode(x)) for x in dataset.src_lines]
        self.tgt_lens = [len(dataset.encode(x)) for x in dataset.tgt_lines]

    def __iter__(self):

        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.pool_size):

            pool = indices[i:i+self.pool_size]

            pool.sort(
                key=lambda x: max(self.src_lens[x], self.tgt_lens[x]),
                reverse=True
            )

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

                    yield batch
                    batch = [idx]
                    max_src = src_len
                    max_tgt = tgt_len

            if batch:
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


def create_token_dataloader(dataset, tokens_per_batch):

    sampler = TokenBatchSampler(dataset, tokens_per_batch)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=CollateFn(dataset.PAD),
        num_workers=2
    )