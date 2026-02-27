# src/inference.py

import torch

def greedy_decode(model, src_tensor, src_lens, pad_idx, bos_idx, eos_idx, max_decoding_len):
    model.eval()

    with torch.no_grad():
        enc_outputs, hidden = model.encoder(src_tensor, src_lens)

        batch_size = src_tensor.size(0)
        inputs = torch.full((batch_size,), bos_idx).to(src_tensor.device)

        outputs = [[] for _ in range(batch_size)]

        for _ in range(max_decoding_len):
            logits, hidden, _ = model.decoder.forward_step(
                inputs, hidden, enc_outputs
            )

            inputs = logits.argmax(1)

            for i in range(batch_size):
                token = inputs[i].item()
                if token == eos_idx:
                    continue
                outputs[i].append(token)

        return outputs


def translate_file(
    model,
    dataset,
    input_lines,
    max_decoding_len,
    device,
    output_path
):
    bos = dataset.BOS
    eos = dataset.EOS
    pad = dataset.PAD
    sp = dataset.sp

    model.eval()

    batch_size = 64
    all_outputs = []

    for i in range(0, len(input_lines), batch_size):
        batch = input_lines[i:i+batch_size]
        encoded = [dataset.encode(x) for x in batch]

        max_len = max(len(x) for x in encoded)

        src_tensor = torch.full(
            (len(encoded), max_len),
            pad
        ).long()

        src_lens = []

        for j, seq in enumerate(encoded):
            src_tensor[j, :len(seq)] = torch.tensor(seq)
            src_lens.append(len(seq))

        src_tensor = src_tensor.to(device)
        src_lens = torch.tensor(src_lens).to(device)

        ids = greedy_decode(
            model,
            src_tensor,
            src_lens,
            pad,
            bos,
            eos,
            max_decoding_len
        )

        for seq in ids:
            filtered = [x for x in seq if x < dataset.vocab_size]
            all_outputs.append(
                sp.decode_ids(filtered) if filtered else ""
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_outputs:
            f.write(line.strip() + "\n")