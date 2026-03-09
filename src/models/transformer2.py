# src/models/transformer2.py

import torch
import torch.nn as nn
from torch import Tensor


class TransformerMT2(nn.Module):
    """
    Максимально сильный Transformer (Pre-LN + современные улучшения).
    Маски возвращаются булевыми (torch.bool) чтобы избежать внутренних конверсий.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_idx: int = 0,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Sinusoidal positional encoding (buffer, float)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))  # shape (1, max_len, d_model)

        # Pre-LN Transformer encoder/decoder (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---------- mask helpers ----------
    def create_mask(self, seq: Tensor) -> Tensor:
        """
        Returns boolean tensor (batch, seq_len): True for VALID tokens (not PAD).
        Kept name 'create_mask' for backward compatibility with previous code.
        """
        return (seq != self.pad_idx).to(torch.bool)

    def _pad_mask_from_valid(self, valid_mask: Tensor) -> Tensor:
        """
        Convert valid_mask (True for valid tokens) -> key_padding_mask (True for PAD positions).
        """
        return (~valid_mask).to(torch.bool)

    def _causal_mask(self, size: int, device: torch.device) -> Tensor:
        """
        Return boolean causal mask (size x size) with True in upper triangle (i < j),
        dtype=torch.bool, suitable for passing as tgt_mask.
        """
        if size <= 0:
            return torch.zeros((0, 0), dtype=torch.bool, device=device)
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)

    # ---------- positional + embedding ----------
    def _add_positional(self, x: Tensor) -> Tensor:
        """
        x: (batch, seq_len) long
        returns: (batch, seq_len, d_model)
        """
        batch, seq_len = x.shape
        pos = self.pos_encoding[:, :seq_len, :].to(x.device)  # type: ignore
        return self.embedding(x) * (self.d_model ** 0.5) + pos

    # ---------- encode / decode ----------
    def encode(self, src: Tensor):
        """
        src: (batch, src_len) integer tensor
        returns:
          memory: (batch, src_len, d_model)
          src_key_padding_mask: (batch, src_len) bool mask (True for PAD positions)
        """
        src = src.long()
        valid = self.create_mask(src)                # True for valid tokens
        src_key_padding_mask = self._pad_mask_from_valid(valid)  # True for PAD
        src_emb = self._add_positional(src)          # (B, S, D)
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode(self, tgt: Tensor, memory: Tensor, src_key_padding_mask: Tensor):
        """
        tgt: (batch, tgt_len) integer tensor (decoder input, typically includes BOS and previous tokens)
        memory: encoder outputs
        src_key_padding_mask: (batch, src_len) bool (True for PAD)
        returns logits: (batch, tgt_len, vocab)
        """
        tgt = tgt.long()
        valid_tgt = self.create_mask(tgt)                # True for valid
        tgt_key_padding_mask = self._pad_mask_from_valid(valid_tgt)  # True for PAD

        tgt_emb = self._add_positional(tgt)              # (B, T, D)
        # causal mask (bool)
        tgt_mask = self._causal_mask(tgt.size(1), device=tgt.device)

        # pass bool masks consistently: tgt_mask (bool), tgt_key_padding_mask (bool), memory_key_padding_mask (bool)
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.fc_out(out)

    # ---------- forward ----------
    def forward(self, src: Tensor, tgt: Tensor):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)  -- includes BOS at pos 0 and EOS at end
        returns logits for positions 1..tgt_len-1 -> shape (batch, tgt_len-1, vocab)
        """
        # ensure long dtype for indexing/embeddings
        src = src.long()
        tgt = tgt.long()

        memory, src_key_padding_mask = self.encode(src)
        logits = self.decode(tgt[:, :-1], memory, src_key_padding_mask)
        return logits