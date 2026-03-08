# src/inference.py

import torch
from pathlib import Path
from typing import List, Optional


@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    dataset,
    max_len: int,
    device: torch.device,
) -> List[List[int]]:
    """
    Universal greedy decode for models implementing .encode() and .decode().
    Returns list of token-id sequences (without BOS, without EOS), ready for sp.decode_ids.
    All masks are boolean inside model; this function keeps everything on device.
    """

    model.eval()
    src = src.to(device)
    batch_size = src.size(0)

    BOS = int(dataset.BOS)
    EOS = int(dataset.EOS)
    vocab_size = int(dataset.vocab_size)

    # encode once
    memory, src_mask = model.encode(src)  # src_mask is bool

    # start tokens
    ys = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)

    outputs: List[List[int]] = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    for _step in range(max_len):
        # decode full current sequence (decoder expects full tgt sequence)
        logits = model.decode(ys, memory, src_mask)  # (B, T, V)
        # limit to real vocab to avoid stray special ids if any
        logits = logits[:, -1, :vocab_size]  # (B, V)
        next_tokens = logits.argmax(dim=-1)  # (B,)

        # append tokens and mark finished
        for i in range(batch_size):
            if not finished[i]:
                t = int(next_tokens[i].item())
                if t == EOS:
                    finished[i] = True
                else:
                    outputs[i].append(t)

        # append to input sequence for next step
        ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)

        if all(finished):
            break

    return outputs


def translate_file(
    model,
    dataset,
    input_lines: List[str],
    max_decoding_len: int,
    device: torch.device,
    output_path: str,
    batch_size: int = 64,
) -> List[str]:
    """
    Translate a list of input_lines using model.greedy decode (universal).
    Produces tokenized (SentencePiece-decoded) text lines via dataset.sp.decode_ids(),
    writes them to output_path and returns list of lines (if return_predictions=True).
    """

    model.eval()
    PAD = int(dataset.PAD)
    predictions: List[str] = []

    for i in range(0, len(input_lines), batch_size):
        batch_lines = input_lines[i : i + batch_size]
        encoded = [dataset.encode(x) for x in batch_lines]  # encode returns list of ids (BOS .. EOS)
        max_src_len = max(len(x) for x in encoded)

        # create src tensor (long) and move to device
        src_tensor = torch.full((len(encoded), max_src_len), PAD, dtype=torch.long)
        for j, seq in enumerate(encoded):
            src_tensor[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        src_tensor = src_tensor.to(device)

        ids_batch = greedy_decode(model, src_tensor, dataset, max_decoding_len, device)

        # decode ids to text via SentencePiece (sp.decode_ids)
        for seq in ids_batch:
            filtered = [int(x) for x in seq if int(x) < dataset.vocab_size]
            text = dataset.sp.decode_ids(filtered) if filtered else ""
            predictions.append(text)

    # write file once
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in predictions:
            f.write(line.strip() + "\n")

    return predictions