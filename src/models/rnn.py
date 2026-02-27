# src/models/rnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size_ext, emb_size, hidden_size, n_layers=1, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size_ext, emb_size, padding_idx=None)
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=dropout if n_layers>1 else 0)
        self.fc = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, src, src_lens):
        # src: (batch, seq)
        emb = self.embedding(src)  # (batch, seq, emb)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # (batch, seq, hidden*2)
        # combine bidirectional hidden
        # hidden: (num_layers*2, batch, hidden)
        # make initial hidden for decoder:
        # concat last layer forward and backward
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        # reshape and combine
        # take last layer
        h_forward = hidden[-2,:,:]
        h_backward = hidden[-1,:,:]
        h = torch.tanh(self.fc(torch.cat((h_forward, h_backward), dim=1)))  # (batch, hidden)
        return outputs, h.unsqueeze(0)  # outputs for attention, hidden for decoder (1, batch, hidden)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.key = nn.Linear(hidden_size*2, hidden_size)  # encoder outputs -> key
        self.query = nn.Linear(hidden_size, hidden_size)

    def forward(self, dec_hidden, enc_outputs, mask=None):
        # dec_hidden: (batch, hidden)
        # enc_outputs: (batch, seq, hidden*2)
        q = self.query(dec_hidden).unsqueeze(1)  # (batch,1,hidden)
        k = self.key(enc_outputs)                 # (batch,seq,hidden)
        scores = torch.bmm(q, k.transpose(1,2)).squeeze(1)  # (batch, seq)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attn = F.softmax(scores, dim=1)  # (batch, seq)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)  # (batch, hidden*2)
        return context, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size_ext, emb_size, hidden_size, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size_ext, emb_size)
        self.rnn = nn.GRU(emb_size + hidden_size*2, hidden_size, batch_first=True)
        self.attn = Attention(hidden_size)
        self.out = nn.Linear(hidden_size + hidden_size*2 + emb_size, vocab_size_ext)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, last_hidden, enc_outputs, mask=None):
        # input_tok: (batch,)  current token ids
        emb = self.embedding(input_tok).unsqueeze(1)  # (batch,1,emb)
        dec_h = last_hidden.squeeze(0)  # (batch,hidden)
        context, attn = self.attn(dec_h, enc_outputs, mask)  # context (batch, hidden*2)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)  # (batch,1, emb+hid*2)
        output, hidden = self.rnn(rnn_input, last_hidden)  # output (batch,1,hidden)
        output = output.squeeze(1)  # (batch, hidden)
        concat_out = torch.cat([output, context, emb.squeeze(1)], dim=1)
        logits = self.out(concat_out)  # (batch, vocab_ext)
        return logits, hidden, attn

    def forward(self, tgt, last_hidden, enc_outputs, mask=None, teacher_forcing_ratio=0.5):
        batch_size = tgt.size(0)
        max_len = tgt.size(1)
        outputs = []
        input_tok = tgt[:,0]  # assume tgt has BOS as first token
        hidden = last_hidden
        for t in range(1, max_len):
            logits, hidden, attn = self.forward_step(input_tok, hidden, enc_outputs, mask)
            outputs.append(logits.unsqueeze(1))
            teacher_force = (torch.rand(1).item() < teacher_forcing_ratio)
            top1 = logits.argmax(1)
            input_tok = tgt[:,t] if teacher_force else top1
        return torch.cat(outputs, dim=1)  # (batch, max_len-1, vocab)

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size_ext, emb_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(vocab_size_ext, emb_size, hidden_size)
        self.decoder = Decoder(vocab_size_ext, emb_size, hidden_size)

    def create_mask(self, src, pad_idx):
        return (src != pad_idx).to(src.device).float()

    def forward(self, src, src_lens, tgt, pad_idx, teacher_forcing_ratio=0.5):
        enc_outputs, enc_hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src, pad_idx)
        dec_outs = self.decoder(tgt, enc_hidden, enc_outputs, mask, teacher_forcing_ratio)
        return dec_outs