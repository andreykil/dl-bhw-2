# src/inference.py
import math
from pathlib import Path
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    dataset,
    max_len: int,
    device: torch.device,
) -> List[List[int]]:
    """
    Батчевый greedy decode (быстрый и простой).
    """
    model.eval()
    src = src.to(device)
    B = src.size(0)
    BOS = int(dataset.BOS)
    EOS = int(dataset.EOS)
    V = int(dataset.vocab_size)

    memory_all, src_mask_all = model.encode(src)  # (B, S, D), (B, S)

    ys = torch.full((B, 1), BOS, dtype=torch.long, device=device)
    outputs: List[List[int]] = [[] for _ in range(B)]
    finished = [False] * B

    for _ in range(max_len):
        logits = model.decode(ys, memory_all, src_mask_all)  # (B, T, V_total)
        logits = logits[:, -1, :V]  # (B, V)
        next_tok = logits.argmax(dim=-1)  # (B,)
        for i in range(B):
            if not finished[i]:
                t = int(next_tok[i].item())
                if t == EOS:
                    finished[i] = True
                else:
                    outputs[i].append(t)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        if all(finished):
            break

    return outputs


# --- утилиты ---

def _length_norm(length: int, alpha: float) -> float:
    return ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)


def _get_banned_ngrams_torch(seqs: torch.Tensor, no_repeat_ngram_size: int):
    """
    Vectorized helper: для каждого seq (shape (..., L)) возвращает для последнего prefix множ. запрещённых токенов.
    Версия НЕ ИДЕАЛЬНАЯ для любых n, но работает быстро для n<=4.
    Возвращает list[set] — оставлено редким (включается только при request).
    (Заметка: полный векторный запрет n-gram сложен; чаще отключают для speed).
    """
    # We will not use heavy vectorized ngram blocking here.
    raise NotImplementedError("Use no_repeat_ngram_size=0 for max speed. For ngram blocking use slower mode.")


# --- быстрый векторизованный beam search ---

@torch.no_grad()
def batched_beam_search_fast(
    model,
    src: torch.Tensor,
    dataset,
    beam_size: int = 5,
    max_len: int = 50,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 0,  # =0 (выкл) для максимальной скорости
    early_stopping: bool = True,
    device: torch.device = torch.device("cpu"),
) -> List[List[int]]:
    """
    Векторизованный (батчевый) beam search — оптимизированная версия.
    Поддерживает `no_repeat_ngram_size=0` (рекомендуется для скорости).
    """

    model.eval()
    src = src.to(device)
    B = src.size(0)
    BOS = int(dataset.BOS)
    EOS = int(dataset.EOS)
    V = int(dataset.vocab_size)

    # encode once
    memory_all, src_mask_all = model.encode(src)  # memory_all: (B, S, D), src_mask_all: (B, S)

    # подготовка повторённой памяти для всех beam гипотез — делаем это один раз
    mem_rep = memory_all.repeat_interleave(beam_size, dim=0)  # (B*beam, S, D)
    mask_rep = src_mask_all.repeat_interleave(beam_size, dim=0)  # (B*beam, S)

    # seqs: (B, beam, L) начальная длина =1 (BOS)
    seqs = torch.full((B, beam_size, 1), BOS, dtype=torch.long, device=device)
    # scores: (B, beam) — logprob sums
    scores = torch.full((B, beam_size), -1e9, device=device)
    scores[:, 0] = 0.0
    # finished flags for beams
    is_finished = torch.zeros((B, beam_size), dtype=torch.bool, device=device)

    completed = [[] for _ in range(B)]  # collect (seq_without_bos, score) tuples

    cur_len = 1

    for step in range(1, max_len + 1):
        # flat seqs to feed decoder
        flat_seqs = seqs.view(B * beam_size, cur_len)  # (B*beam, cur_len)

        # decode over all beams at once
        logits = model.decode(flat_seqs, mem_rep, mask_rep)  # (B*beam, cur_len, V_total)
        # last step logprobs
        logp = F.log_softmax(logits[:, -1, :V], dim=-1)  # (B*beam, V)
        logp = logp.view(B, beam_size, V)  # (B, beam, V)

        # prevent expanding finished beams: set logp to -inf except EOS which we set to 0
        if is_finished.any():
            finished_mask = is_finished.unsqueeze(-1)  # (B, beam, 1)
            logp = logp.masked_fill(finished_mask, -1e9)
            # allow EOS by leaving 0 delta (effectively keeps same score)
            logp[..., EOS] = logp[..., EOS].masked_fill(finished_mask.squeeze(-1), 0.0)

        # no_repeat_ngram_size — опция: если >0, будем делать более медленное (но всё ещё векторизованное) обработание
        if no_repeat_ngram_size > 0:
            # Разрешаем медленную обработку: реализуем проверку на CPU для каждой beam (можно оптимизировать далее).
            # Здесь мы всё ещё пытаемся минимизировать копии: извлечём seqs в CPU единоразово.
            seqs_cpu = seqs.cpu().numpy()  # shape (B, beam, cur_len)
            for b in range(B):
                for k in range(beam_size):
                    seq_no_bos = seqs_cpu[b, k, 1:].tolist() if cur_len > 1 else []
                    # build banned set (Python) — это медленная часть, но её можно отключать
                    if len(seq_no_bos) + 1 < no_repeat_ngram_size:
                        continue
                    # collect existing ngrams
                    n = no_repeat_ngram_size
                    forb = {}
                    ln = len(seq_no_bos)
                    for i in range(ln - n + 1):
                        prefix = tuple(seq_no_bos[i:i + n - 1])
                        next_tok = seq_no_bos[i + n - 1]
                        forb.setdefault(prefix, set()).add(next_tok)
                    cur_prefix = tuple(seq_no_bos[-(n - 1):]) if n - 1 > 0 else tuple()
                    banned = forb.get(cur_prefix, None)
                    if banned:
                        # mask banned tokens
                        idxs = torch.tensor(list(banned), dtype=torch.long, device=device)
                        logp[b, k, idxs] = -1e9

        # compute candidate scores
        total = scores.unsqueeze(-1) + logp  # (B, beam, V)
        flat = total.view(B, beam_size * V)  # (B, beam*V)

        topk_scores, topk_idx = torch.topk(flat, k=beam_size, dim=-1)  # (B, beam)
        prev_beam = topk_idx // V  # (B, beam)
        token_idx = topk_idx % V   # (B, beam)

        # gather previous sequences for selected prev_beam
        # prev_seqs: (B, beam, cur_len)
        prev_seqs = seqs
        # build index for gather: (B, beam, cur_len)
        prev_beam_idx = prev_beam.unsqueeze(-1).expand(-1, -1, cur_len)
        selected_prev = torch.gather(prev_seqs, 1, prev_beam_idx)  # (B, beam, cur_len)

        # create new seqs by appending token_idx
        new_seqs = torch.cat([selected_prev, token_idx.unsqueeze(-1)], dim=2)  # (B, beam, cur_len+1)
        new_scores = topk_scores  # (B, beam)

        # update finished flags: gather prev is_finished and OR with token==EOS
        prev_finished = torch.gather(is_finished, 1, prev_beam)  # (B, beam)
        new_finished = prev_finished | (token_idx == EOS)

        # add completed sequences where token==EOS (collect CPU small list)
        eos_mask = (token_idx == EOS)
        if eos_mask.any():
            # move minimal data to CPU to append to Python lists
            new_seqs_cpu = new_seqs.cpu().numpy()
            new_scores_cpu = new_scores.cpu().numpy()
            eos_mask_cpu = eos_mask.cpu().numpy()
            for b in range(B):
                for k in range(beam_size):
                    if eos_mask_cpu[b, k]:
                        seq_tokens = new_seqs_cpu[b, k].tolist()
                        # remove BOS if present (first token)
                        if seq_tokens and seq_tokens[0] == BOS:
                            seq_no_bos = seq_tokens[1:]
                        else:
                            seq_no_bos = seq_tokens[:]
                        # drop trailing EOS
                        if seq_no_bos and seq_no_bos[-1] == EOS:
                            seq_no_bos = seq_no_bos[:-1]
                        completed[b].append((seq_no_bos, float(new_scores_cpu[b, k])))

        # assign variables for next step
        seqs = new_seqs.to(device)
        scores = new_scores.to(device)
        is_finished = new_finished.to(device)
        cur_len += 1

        # early stopping check (vectorized-ish)
        if early_stopping:
            all_stop = True
            # check per batch
            for b in range(B):
                if len(completed[b]) < 1:
                    all_stop = False
                    break
                # worst completed norm
                worst_norm = min(sc / _length_norm(len(s), length_penalty) for s, sc in completed[b])
                # best live possible normalized (heuristic)
                best_live_norm = float(scores[b].max().item()) / _length_norm(cur_len - 1, length_penalty)
                if best_live_norm > worst_norm:
                    all_stop = False
                    break
            if all_stop:
                break

        # if no live beams remain -> break
        if (~is_finished).sum().item() == 0:
            break

    # finalize results
    results: List[List[int]] = []
    for b in range(B):
        # select best from completed by normalized score
        if len(completed[b]) == 0:
            # fallback: choose best live
            idx = int(scores[b].argmax().item())
            seq_tokens = seqs[b, idx].tolist()
            if seq_tokens and seq_tokens[0] == BOS:
                seq_tokens = seq_tokens[1:]
            if seq_tokens and seq_tokens[-1] == EOS:
                seq_tokens = seq_tokens[:-1]
            results.append(seq_tokens)
        else:
            best_seq = None
            best_norm = -1e9
            for seq_tokens, sc in completed[b]:
                norm = sc / _length_norm(len(seq_tokens), length_penalty)
                if norm > best_norm:
                    best_norm = norm
                    best_seq = seq_tokens
            results.append(best_seq if best_seq is not None else [])

    return results


# ---------------- translate_file ----------------

def translate_file(
    model,
    dataset,
    input_lines: List[str],
    max_decoding_len: int,
    device: torch.device,
    output_path: str,
    batch_size: int = 64,
    mode: str = "beam",
    beam_size: int = 5,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 0,
    early_stopping: bool = True,
) -> List[str]:
    """
    Переводит input_lines, записывает в файл и возвращает predictions.
    mode: 'beam' (векторизованный быстрый beam) или 'greedy'
    Рекомендуется: no_repeat_ngram_size=0 для максимальной скорости.
    """
    assert mode in ("beam", "greedy")
    model.eval()
    PAD = int(dataset.PAD)
    predictions: List[str] = []

    for i in range(0, len(input_lines), batch_size):
        batch_lines = input_lines[i : i + batch_size]
        encoded = [dataset.encode(x) for x in batch_lines]
        max_src_len = max(len(seq) for seq in encoded)

        src_tensor = torch.full((len(encoded), max_src_len), PAD, dtype=torch.long)
        for j, seq in enumerate(encoded):
            src_tensor[j, : len(seq)] = torch.tensor(seq, dtype=torch.long)

        src_tensor = src_tensor.to(device)

        if mode == "greedy":
            ids_batch = greedy_decode(model, src_tensor, dataset, max_decoding_len, device)
        else:
            ids_batch = batched_beam_search_fast(
                model=model,
                src=src_tensor,
                dataset=dataset,
                beam_size=beam_size,
                max_len=max_decoding_len,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                device=device,
            )

        for seq in ids_batch:
            filtered = [int(x) for x in seq if int(x) < dataset.vocab_size]
            text = dataset.sp.decode_ids(filtered) if filtered else ""
            predictions.append(text)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in predictions:
            f.write(line.strip() + "\n")

    return predictions