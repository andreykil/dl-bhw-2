# src/inference.py
import torch
from pathlib import Path
from typing import List, Optional
import math

# ---------------------------
# Helpers
# ---------------------------

def _get_banned_tokens_for_ngrams(hyp_tokens: List[int], n: int) -> set:
    """
    Возвращает множество токенов, которые приведут к повтору n-грам при добавлении их в конец hyp_tokens.
    hyp_tokens - список токенов (идов), включая BOS.
    n - размер n-gram, если <=1 возвращается пустое множество.
    """
    if n <= 1 or len(hyp_tokens) + 1 < n:
        return set()
    banned = set()
    # last (n-1) tokens
    prefix = tuple(hyp_tokens[-(n-1):])
    for i in range(len(hyp_tokens) - (n - 1)):
        if tuple(hyp_tokens[i:i + (n - 1)]) == prefix and (i + n - 1) < len(hyp_tokens):
            banned.add(hyp_tokens[i + n - 1])
    return banned


# ---------------------------
# Greedy decode (batched)
# ---------------------------

@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    dataset,
    max_len: int,
    device: torch.device,
    no_repeat_ngram_size: int = 0
) -> List[List[int]]:
    """
    Батчевый greedy decode.
    Возвращает список списков токен-идов (без BOS и без EOS), готовых к sp.decode_ids.
    """
    model.eval()
    src = src.to(device)
    B = src.size(0)

    BOS = int(dataset.BOS)
    EOS = int(dataset.EOS)
    PAD = int(dataset.PAD)
    base_vocab = int(dataset.vocab_size)

    with torch.inference_mode():
        memory, src_mask = model.encode(src)  # модель должна возвращать (memory, src_mask)

        ys = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        outputs: List[List[int]] = [[] for _ in range(B)]
        finished = [False] * B
        hyps = [[BOS] for _ in range(B)]  # для ngram blocking

        for _step in range(max_len):
            logits = model.decode(ys, memory, src_mask)  # (B, T, V_total)
            last_logits = logits[:, -1, :]               # (B, V_total)
            logprobs = torch.log_softmax(last_logits, dim=-1)

            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                # применяем запрет n-грам (побатчно)
                lp = logprobs.clone()
                V = lp.size(-1)
                for i in range(B):
                    if finished[i]:
                        continue
                    banned = _get_banned_tokens_for_ngrams(hyps[i], no_repeat_ngram_size)
                    if banned:
                        banned_filtered = [t for t in banned if 0 <= t < V]
                        if banned_filtered:
                            lp[i, banned_filtered] = -1e9
                next_tokens = lp.argmax(dim=-1)
            else:
                next_tokens = logprobs.argmax(dim=-1)

            # обработка токенов
            for i in range(B):
                if finished[i]:
                    continue
                t = int(next_tokens[i].item())
                if t == EOS:
                    finished[i] = True
                else:
                    outputs[i].append(t)
                    hyps[i].append(t)

            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)

            if all(finished):
                break

    return outputs


# ---------------------------
# Beam search (batched, vectorized)
# ---------------------------

@torch.no_grad()
def beam_search_batch(
    model,
    src: torch.Tensor,
    dataset,
    max_len: int,
    device: torch.device,
    beam_size: int = 5,
    no_repeat_ngram_size: int = 0,
    length_penalty: float = 0.6,
) -> List[List[int]]:
    """
    Батчевый, векторизованный beam search.
    Возвращает список списков токен-идов (без BOS и без EOS).
    Поддерживает length_penalty и (опционально) no_repeat_ngram_size.
    """
    model.eval()
    src = src.to(device)
    B = src.size(0)

    BOS = int(dataset.BOS)
    EOS = int(dataset.EOS)
    PAD = int(dataset.PAD)

    with torch.inference_mode():
        memory, src_mask = model.encode(src)  # memory: (B, S, E)
        # expand memory for beams: (B*beam, S, E)
        memory = memory.unsqueeze(1).expand(B, beam_size, -1, -1).contiguous().view(B * beam_size, memory.size(-2), memory.size(-1))
        if src_mask is not None:
            # src_mask expected shape (B, S) bool; expand similarly -> (B*beam, S)
            src_mask = src_mask.unsqueeze(1).expand(B, beam_size, -1).contiguous().view(B * beam_size, -1)

        # initial sequences' scores: only first beam active
        scores = torch.full((B, beam_size), -1e9, device=device)
        scores[:, 0] = 0.0
        # finished flags per (B,beam)
        finished = [[False] * beam_size for _ in range(B)]

        # ys stores token sequences for every hypothesis as rows: shape (B*beam, cur_len)
        ys = torch.full((B * beam_size, 1), BOS, dtype=torch.long, device=device)

        for step in range(max_len):
            # decode all current hypotheses at once
            logits = model.decode(ys, memory, src_mask)      # (B*beam, T, V_total)
            last_logits = logits[:, -1, :]                   # (B*beam, V_total)
            V_total = last_logits.size(-1)

            # convert to (B, beam, V)
            logprobs = torch.log_softmax(last_logits, dim=-1).view(B, beam_size, V_total)  # (B, beam, V)

            # apply ngram banning if needed (small python loop)
            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                for b in range(B):
                    for k in range(beam_size):
                        if finished[b][k]:
                            continue
                        hyp_row = ys[b * beam_size + k].tolist()
                        banned = _get_banned_tokens_for_ngrams(hyp_row, no_repeat_ngram_size)
                        if banned:
                            banned_list = [t for t in banned if 0 <= t < V_total]
                            if banned_list:
                                logprobs[b, k, banned_list] = -1e9

            # For beams that are already finished, prevent expansion:
            # set their logprobs to -inf except set EOS logprob to 0 so score doesn't change
            
            finished_mask = torch.as_tensor(finished, dtype=torch.bool, device=device)

            if finished_mask.any():

                # запретить все токены
                fm = finished_mask.unsqueeze(-1).expand(-1, -1, V_total)
                logprobs = logprobs.masked_fill(fm, -1e9)

                # разрешить EOS
                if 0 <= EOS < V_total:
                    idx = finished_mask.nonzero(as_tuple=False)  # (N,2)
                    if idx.numel() > 0:
                        b_idx = idx[:,0]
                        beam_idx = idx[:,1]

                        logprobs[b_idx, beam_idx, EOS] = 0.0

            # For each beam, take top-k token candidates (k = beam_size)
            # topk_vals, topk_idx shapes: (B, beam, k)
            topk_vals, topk_idx = torch.topk(logprobs, k=beam_size, dim=-1)

            # combined candidate scores = prev_scores.unsqueeze(-1) + topk_vals  => (B, beam, k)
            combined = scores.unsqueeze(-1) + topk_vals  # (B, beam, k)

            # flatten candidates per batch: (B, beam*k)
            Bk = beam_size * beam_size
            combined_flat = combined.view(B, Bk)

            # pick top beam_size candidates overall per batch
            topk_comb_vals, topk_comb_idx = combined_flat.topk(k=beam_size, dim=1)  # (B, beam)

            # decode parent beam and candidate position
            k_val = beam_size
            parent_beam_idx = (topk_comb_idx // k_val)   # (B, beam) indices in [0..beam-1]
            pos_in_topk = (topk_comb_idx % k_val)        # (B, beam) indices in [0..k-1]

            # gather selected token ids: topk_idx[b, parent_beam_idx[b,k], pos_in_topk[b,k]]
            # We build indexing tensors
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, beam_size)  # (B, beam)
            token_ids = topk_idx[batch_idx, parent_beam_idx, pos_in_topk]  # (B, beam) token ids
            new_scores = topk_comb_vals                                   # (B, beam) combined scores

            # Build next Ys: select parent rows from ys and append corresponding token_ids
            # parent_flat_indices = (b * beam_size + parent_beam_idx[b,k]).flatten()
            parent_base = (torch.arange(B, device=device) * beam_size).unsqueeze(1)  # (B,1)
            parent_flat = (parent_base + parent_beam_idx).view(-1)                  # (B*beam,)
            # select parent rows
            ys_selected = ys[parent_flat]  # (B*beam, cur_len)
            # tokens to append
            token_flat = token_ids.view(-1).unsqueeze(1)  # (B*beam,1)
            # append
            ys = torch.cat([ys_selected, token_flat], dim=1)  # (B*beam, cur_len+1)

            # update finished flags: finished[b][k] = True if token == EOS or parent was finished
            # parent finished flags extraction
            parent_finished = []
            for b in range(B):
                parent_finished.extend([finished[b][int(idx)] for idx in parent_beam_idx[b].tolist()])
            # convert into 2d
            new_finished = []
            idx_pf = 0
            for b in range(B):
                row = []
                for k in range(beam_size):
                    was_finished = parent_finished[idx_pf]
                    idx_pf += 1
                    is_eos_now = (int(token_ids[b, k].item()) == EOS)
                    row.append(was_finished or is_eos_now)
                new_finished.append(row)
            finished = new_finished

            # update scores
            scores = new_scores

            # early stopping if all beams finished
            all_finished = True
            for b in range(B):
                if not all(finished[b]):
                    all_finished = False
                    break
            if all_finished:
                break

        # end for step

        # choose best hypothesis per batch using length penalty
        # ys currently shape (B*beam, cur_len_final); reshape to (B, beam, L)
        ys = ys.view(B, beam_size, -1)  # (B, beam, L)
        results: List[List[int]] = []
        for b in range(B):
            best_score = -1e18
            best_seq: List[int] = []
            for k in range(beam_size):
                seq_row = ys[b, k].tolist()
                # remove BOS and tokens after EOS
                if EOS in seq_row:
                    seq_trim = seq_row[1: seq_row.index(EOS)]
                else:
                    seq_trim = seq_row[1:]
                length = max(1, len(seq_trim))
                if length_penalty and length_penalty > 0:
                    lp = ((5.0 + length) / 6.0) ** length_penalty
                else:
                    lp = 1.0
                sc = scores[b, k].item() / lp if lp != 0 else scores[b, k].item()
                if sc > best_score:
                    best_score = sc
                    best_seq = seq_trim
            results.append(best_seq)

    return results


# ---------------------------
# translate_file wrapper
# ---------------------------

def translate_file(
    model,
    dataset,
    input_lines: List[str],
    max_decoding_len: int,
    device: torch.device,
    output_path: Optional[str] = None,
    batch_size: int = 64,
    mode: str = "greedy",
    beam_size: int = 5,
    no_repeat_ngram_size: int = 0,
    length_penalty: float = 0.6,
) -> List[str]:
    """
    Перевод списка строк через модель.
    Параметры:
      - model: модель с интерфейсом encode/decode
      - dataset: TranslationDataset (имеет .encode(text) -> list[int], .sp.decode_ids)
      - input_lines: список строк на source
      - max_decoding_len: int
      - device: torch.device
      - output_path: если None — не записывать файл; иначе записать
      - batch_size, mode, beam_size, no_repeat_ngram_size, length_penalty
    Возвращает список переводов (строк), в том числе и если output_path=None.
    """
    model.eval()
    all_outputs: List[str] = []
    PAD = int(dataset.PAD)
    sp = dataset.sp

    # inference loop per batch
    for i in range(0, len(input_lines), batch_size):
        batch_lines = input_lines[i:i + batch_size]
        encoded = [dataset.encode(x) for x in batch_lines]  # encode includes BOS/EOS as we defined
        max_src_len = max(len(s) for s in encoded)
        src_tensor = torch.full((len(encoded), max_src_len), PAD, dtype=torch.long)
        for j, seq in enumerate(encoded):
            src_tensor[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        src_tensor = src_tensor.to(device)

        if mode == "greedy":
            ids_batch = greedy_decode(
                model=model,
                src=src_tensor,
                dataset=dataset,
                max_len=max_decoding_len,
                device=device,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
        elif mode == "beam":
            ids_batch = beam_search_batch(
                model=model,
                src=src_tensor,
                dataset=dataset,
                max_len=max_decoding_len,
                device=device,
                beam_size=beam_size,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty
            )
        else:
            raise ValueError("mode must be 'greedy' or 'beam'")

        # decode ids -> text; filter special tokens (ids >= dataset.vocab_size)
        for seq in ids_batch:
            filtered = [int(x) for x in seq if int(x) < dataset.vocab_size]
            text = sp.decode_ids(filtered) if filtered else ""
            all_outputs.append(text)

    # optionally write to file
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for line in all_outputs:
                f.write(line.strip() + "\n")

    return all_outputs