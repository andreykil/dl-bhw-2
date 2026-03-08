# src/inference.py

import torch
from pathlib import Path
from typing import List


@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    dataset,
    max_len: int,
    device: torch.device,
):
    """
    Универсальный greedy decoding.
    Работает со всеми моделями с .encode() и .decode()
    """

    model.eval()

    BOS = dataset.BOS
    EOS = dataset.EOS

    batch_size = src.shape[0]

    memory, src_mask = model.encode(src.to(device))

    ys = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)

    outputs = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    for _ in range(max_len):

        logits = model.decode(ys, memory, src_mask)
        logits = logits[:, -1, :dataset.vocab_size]
        
        next_token = logits.argmax(-1)

        for i in range(batch_size):

            if not finished[i]:

                token = int(next_token[i].item())

                if token == EOS:
                    finished[i] = True
                else:
                    if token < dataset.vocab_size:
                        outputs[i].append(token)

        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

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
):

    model.eval()

    predictions = []

    for i in range(0, len(input_lines), batch_size):

        batch_lines = input_lines[i : i + batch_size]

        encoded = [dataset.encode(x) for x in batch_lines]

        max_src_len = max(len(x) for x in encoded)

        src_tensor = torch.full(
            (len(encoded), max_src_len),
            dataset.PAD,
            dtype=torch.long
        )

        for j, seq in enumerate(encoded):
            src_tensor[j, :len(seq)] = torch.tensor(seq)

        src_tensor = src_tensor.to(device)

        ids_batch = greedy_decode(
            model,
            src_tensor,
            dataset,
            max_decoding_len,
            device
        )

        for seq in ids_batch:

            filtered = [x for x in seq if x < dataset.vocab_size]

            text = dataset.sp.decode_ids(filtered) if filtered else ""

            predictions.append(text)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:

        for line in predictions:
            f.write(line.strip() + "\n")

    return predictions