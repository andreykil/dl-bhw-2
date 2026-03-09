# src/models/transformer.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


class TransformerMT(nn.Module):
    """
    Clean, efficient Transformer wrapper with explicit boolean masks.
    Provides:
      - encode(src) -> (memory, src_key_padding_mask)
      - decode(tgt, memory, src_mask) -> logits (batch, T, vocab)
      - forward(src, src_lens, tgt) -> logits for training (batch, T, vocab)
    """

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

        # token + positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.pos_embedding = nn.Embedding(max_len, emb_size)

        # transformer: batch_first=True so inputs are (B, S, D)
        # norm_first often gives slightly better stability (optional)
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )

        self.output_layer = nn.Linear(emb_size, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -------------------------
    # Mask helpers (return bool masks)
    # -------------------------
    def create_padding_mask(self, seq: Tensor) -> Tensor:
        """
        seq: (batch, seq_len) of token ids (any integer dtype)
        returns: bool mask (batch, seq_len) where True indicates PAD.
        """
        # ensure boolean dtype
        return (seq == self.pad_idx).to(torch.bool)

    def create_subsequent_mask(self, size: int, device: torch.device) -> Tensor:
        """
        Causal mask for decoding: shape (size, size), True in upper triangle (i<j),
        dtype=torch.bool, suitable as attn_mask in modern PyTorch.
        """
        if size <= 0:
            return torch.zeros((0, 0), dtype=torch.bool, device=device)
        # use bool mask directly to avoid float->bool conversions
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    # -------------------------
    # Positional + token embeddings
    # -------------------------
    def add_positional(self, tokens: Tensor) -> Tensor:
        """
        tokens: (batch, seq_len) long tensor of token ids
        returns embeddings: (batch, seq_len, emb_size)
        """
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        return self.token_embedding(tokens) + self.pos_embedding(positions)

    # -------------------------
    # Encode / Decode API
    # -------------------------
    def encode(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        """
        src: (batch, src_len) long tensor
        returns:
          memory: (batch, src_len, emb)
          src_key_padding_mask: (batch, src_len) bool
        """
        src = src.long()
        src_key_padding_mask = self.create_padding_mask(src)  # bool
        src_emb = self.add_positional(src)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, tgt: Tensor, memory: Tensor, src_mask: Tensor) -> Tensor:
        """
        tgt: (batch, tgt_len) long tensor of tokens (for decoding input)
        memory: encoder outputs
        src_mask: src_key_padding_mask (batch, src_len) bool
        returns logits: (batch, tgt_len, vocab)
        """
        tgt = tgt.long()
        tgt_emb = self.add_positional(tgt)

        # boolean causal mask and padding mask
        tgt_mask = self.create_subsequent_mask(tgt.shape[1], device=tgt.device)  # (T, T) bool
        tgt_key_padding_mask = self.create_padding_mask(tgt)                      # (B, T) bool

        # pass bool masks; modern PyTorch accepts bool masks
        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_mask
        )
        logits = self.output_layer(out)
        return logits

    # -------------------------
    # Forward for training
    # -------------------------
    def forward(self, src: Tensor, tgt: Tensor, pad_idx: Optional[int] = None) -> Tensor:
        """
        Training forward: expects tgt with BOS at pos 0; returns logits for positions 1..len(tgt)-1
        logits shape: (batch, tgt_len-1, vocab)
        """
        memory, src_mask = self.encode(src)
        tgt_input = tgt[:, :-1]  # drop last token (EOS) as input to decoder
        logits = self.decode(tgt_input, memory, src_mask)
        return logits