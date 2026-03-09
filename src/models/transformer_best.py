# src/models/transformer_best.py
import torch
import torch.nn as nn


class TransformerBest(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        pad_idx=0,
        max_len=512
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        # positional encoding
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len).unsqueeze(1)

        div = torch.exp(
            torch.arange(0,d_model,2) *
            -(torch.log(torch.tensor(10000.0))/d_model)
        )

        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)

        self.register_buffer("pos",pe.unsqueeze(0))

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.out = nn.Linear(d_model,vocab_size,bias=False)

        # weight tying
        self.out.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):

        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def pad_mask(self,x):
        return (x==self.pad_idx)

    def causal_mask(self,size,device):

        return torch.triu(
            torch.ones(size,size,device=device,dtype=torch.bool),
            diagonal=1
        )

    def embed(self,x):

        seq = x.size(1)

        return (
            self.embedding(x)*(self.d_model**0.5) + self.pos[:,:seq] # type: ignore
        )

    def encode(self,src):

        src_mask = self.pad_mask(src)

        src = self.embed(src)

        memory = self.transformer.encoder(
            src,
            src_key_padding_mask=src_mask
        )

        return memory,src_mask

    def decode(self,tgt,memory,src_mask):

        tgt_mask = self.causal_mask(
            tgt.size(1),
            tgt.device
        )

        tgt_pad = self.pad_mask(tgt)

        tgt = self.embed(tgt)

        out = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_mask
        )

        return self.out(out)

    def forward(self, src, tgt):

        memory,src_mask = self.encode(src)

        tgt_in = tgt[:,:-1]

        logits = self.decode(
            tgt_in,
            memory,
            src_mask
        )

        return logits