# src/models/transformer.py

import torch
import torch.nn as nn
from torch import Tensor


class TransformerMT(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        emb_size: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, emb_size)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer = nn.Linear(emb_size, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq: Tensor) -> Tensor:
        return (seq == self.pad_idx)

    def create_subsequent_mask(self, size, device):
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device),
            diagonal=1
        )

    def add_positional(self, tokens: Tensor) -> Tensor:
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, -1)
        return self.token_embedding(tokens) + self.pos_embedding(positions)

    def encode(self, src: Tensor):
        src_mask = self.create_padding_mask(src)
        src_emb = self.add_positional(src)

        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_mask
        )

        return memory, src_mask

    def decode(self, tgt: Tensor, memory: Tensor, src_mask: Tensor):
        tgt_emb = self.add_positional(tgt)

        tgt_mask = self.create_subsequent_mask(tgt.shape[1], tgt.device)
        tgt_padding_mask = self.create_padding_mask(tgt)

        out = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_mask
        )

        logits = self.output_layer(out)

        return logits

    def forward(self, src: Tensor, src_lens: Tensor, tgt: Tensor, pad_idx=None):

        if pad_idx is None:
            pad_idx = self.pad_idx

        memory, src_mask = self.encode(src)

        tgt_input = tgt[:, :-1]

        logits = self.decode(tgt_input, memory, src_mask)

        return logits