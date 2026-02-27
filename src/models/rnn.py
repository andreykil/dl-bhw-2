# src/models/rnn.py (замените Encoder, Decoder, Seq2Seq на эту версию)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size_ext, emb_size, hidden_size, pad_idx, n_layers=1, dropout=0.3):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size_ext, emb_size, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            emb_size,
            hidden_size,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        # проектируем concat(forward, backward) -> hidden
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src, src_lens):
        # src: (batch, seq)
        emb = self.embedding(src)  # (batch, seq, emb)
        emb = self.emb_dropout(emb)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs_packed, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs_packed, batch_first=True)
        # hidden: (num_layers * num_directions, batch, hidden)
        # for bidirectional: num_directions = 2 => shape (n_layers*2, B, H)
        # Need to combine per-layer forward/backward -> (n_layers, B, H)
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        # hidden is tensor of shape (n_layers*2, batch, hidden)
        # reorganize: for i in 0..n_layers-1 take forward = hidden[2*i], backward = hidden[2*i+1]
        n_layers = self.n_layers
        batch = hidden.size(1)
        H = hidden.size(2)
        # collect layers
        new_hidden = []
        for i in range(n_layers):
            h_fwd = hidden[2 * i]      # (B, H)
            h_bwd = hidden[2 * i + 1]  # (B, H)
            h_cat = torch.cat((h_fwd, h_bwd), dim=1)  # (B, 2H)
            h_proj = torch.tanh(self.fc(h_cat))       # (B, H)
            new_hidden.append(h_proj.unsqueeze(0))    # list of (1, B, H)

        # stack -> (n_layers, B, H)
        final_hidden = torch.cat(new_hidden, dim=0)

        return outputs, final_hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.key = nn.Linear(hidden_size * 2, hidden_size)  # enc outputs -> key
        self.query = nn.Linear(hidden_size, hidden_size)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # dec_hidden: (batch, hidden)
        # enc_outputs: (batch, seq, hidden*2)
        q = self.query(dec_hidden).unsqueeze(1)  # (batch,1,hidden)
        k = self.key(enc_outputs)                 # (batch,seq,hidden)
        scores = torch.bmm(q, k.transpose(1, 2)).squeeze(1)  # (batch, seq)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=1)  # (batch, seq)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)  # (batch, hidden*2)
        return context, attn


class Decoder(nn.Module):
    def __init__(self, vocab_size_ext, emb_size, hidden_size, pad_idx, n_layers=1, dropout=0.3):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size_ext, emb_size, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_size + hidden_size * 2, hidden_size, num_layers=n_layers, batch_first=True)
        self.attn = Attention(hidden_size)
        self.out = nn.Linear(hidden_size + hidden_size * 2 + emb_size, vocab_size_ext)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, last_hidden, enc_outputs, mask=None):
        # input_tok: (batch,)  current token ids
        emb = self.embedding(input_tok).unsqueeze(1)  # (batch,1,emb)
        emb = self.emb_dropout(emb)
        # dec_h to compute attention: use the last layer's hidden state
        dec_h = last_hidden[-1]   # (batch, hidden)
        context, attn = self.attn(dec_h, enc_outputs, mask)  # context (batch, hidden*2)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)  # (batch,1, emb+hid*2)
        output, hidden = self.rnn(rnn_input, last_hidden)  # output (batch,1,hidden); hidden: (n_layers,B,H)
        output = output.squeeze(1)  # (batch, hidden)
        concat_out = torch.cat([output, context, emb.squeeze(1)], dim=1)
        logits = self.out(concat_out)  # (batch, vocab_ext)
        return logits, hidden, attn

    def forward(self, tgt, last_hidden, enc_outputs, mask=None, teacher_forcing_ratio=0.5):
        batch_size = tgt.size(0)
        max_len = tgt.size(1)
        outputs = []
        input_tok = tgt[:, 0]  # assume tgt has BOS as first token
        hidden = last_hidden
        for t in range(1, max_len):
            logits, hidden, attn = self.forward_step(input_tok, hidden, enc_outputs, mask)
            outputs.append(logits.unsqueeze(1))
            teacher_force = (torch.rand(1).item() < teacher_forcing_ratio)
            top1 = logits.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1
        return torch.cat(outputs, dim=1)  # (batch, max_len-1, vocab)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size_ext, emb_size, hidden_size, pad_idx, n_layers=1, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(vocab_size_ext, emb_size, hidden_size, pad_idx, n_layers=n_layers, dropout=dropout)
        self.decoder = Decoder(vocab_size_ext, emb_size, hidden_size, pad_idx, n_layers=n_layers, dropout=dropout)

    def create_mask(self, src, pad_idx):
        return (src != pad_idx).to(src.device)

    def forward(self, src, src_lens, tgt, pad_idx, teacher_forcing_ratio=0.5):
        enc_outputs, enc_hidden = self.encoder(src, src_lens)   # enc_hidden: (n_layers, B, H)
        mask = self.create_mask(src, pad_idx)
        dec_outs = self.decoder(tgt, enc_hidden, enc_outputs, mask, teacher_forcing_ratio)
        return dec_outs