# src/inference.py
import math
from pathlib import Path
from typing import List
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
    """Быстрый батчевый greedy decode (возвращает токены без BOS/EOS)."""
    model.eval()
    src = src.to(device)
    B = src.size(0)
    BOS = int(dataset.BOS)
    EOS = int(dataset.EOS)
    V = int(dataset.vocab_size)

    memory_all, src_mask_all = model.encode(src)  # (B,S,D), (B,S) bool

    ys = torch.full((B, 1), BOS, dtype=torch.long, device=device)
    outputs = [[] for _ in range(B)]
    finished = [False] * B

    for _ in range(max_len):
        logits = model.decode(ys, memory_all, src_mask_all)  # (B, T, Vtotal)
        logits = logits[:, -1, :V]
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


def _length_norm(length: int, alpha: float) -> float:
    return ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)


@torch.no_grad()
def batched_beam_search_optimized(
    model,
    src: torch.Tensor,
    dataset,
    beam_size: int = 5,
    max_len: int = 50,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 0,
    early_stopping: bool = True,
    device: torch.device = torch.device("cpu"),
) -> List[List[int]]:
    """
    Эффективный батчевый beam search (векторизованный).
    Возвращает для каждого примера список токенов (без BOS и без EOS).
    Рекомендуется no_repeat_ngram_size=0 для скорости.
    """
    model.eval()
    src = src.to(device)
    B = src.size(0)
    BOS = int(dataset.BOS)
    EOS = int(dataset.EOS)
    V = int(dataset.vocab_size)

    # Энкодим один раз
    memory_all, src_mask_all = model.encode(src)  # (B, S, D), (B, S)

    # Раздуваем память один раз для всех beam гипотез
    mem_rep = memory_all.repeat_interleave(beam_size, dim=0)  # (B*beam, S, D)
    mask_rep = src_mask_all.repeat_interleave(beam_size, dim=0)  # (B*beam, S)

    # seqs: (B, beam, L) начальный L=1 (BOS)
    seqs = torch.full((B, beam_size, 1), BOS, dtype=torch.long, device=device)
    scores = torch.full((B, beam_size), -1e9, device=device)
    scores[:, 0] = 0.0
    is_finished = torch.zeros((B, beam_size), dtype=torch.bool, device=device)

    # Для хранения лучших завершённых гипотез векторно
    best_completed_score = torch.full((B,), float("-1e9"), device=device)  # нормализованная оценка
    # будем хранить best completed sequence (паддированную до max_len без BOS)
    best_completed_seq = torch.full((B, max_len), dataset.PAD, dtype=torch.long, device=device)

    cur_len = 1  # текущее количество токенов в seqs (включая BOS)

    for step in range(1, max_len + 1):
        flat_seqs = seqs.view(B * beam_size, cur_len)  # (B*beam, cur_len)
        # decode единожды
        logits = model.decode(flat_seqs, mem_rep, mask_rep)  # (B*beam, cur_len, Vtotal)
        logp = F.log_softmax(logits[:, -1, :V], dim=-1)  # (B*beam, V)
        logp = logp.view(B, beam_size, V)  # (B, beam, V)

        # запретить расширение для уже завершенных beam:
        if is_finished.any():
            finished_mask = is_finished.unsqueeze(-1)  # (B, beam, 1)
            logp = logp.masked_fill(finished_mask, -1e9)
            # позволить EOS (оставляем 0 delta)
            logp[..., EOS] = logp[..., EOS].masked_fill(finished_mask.squeeze(-1), 0.0)

        # no_repeat_ngram: по умолчанию выключено (0). Векторная реализация сложна и замедляет.
        if no_repeat_ngram_size > 0:
            # Более быстрая стратегия: вычисляем бан-листы для текущих seqs на CPU один раз за шаг
            # и маскируем logp. Это медленнее, чем отключение, но корректно.
            seqs_cpu = seqs.cpu().tolist()  # shape (B, beam, cur_len)
            for b in range(B):
                for k in range(beam_size):
                    seq_no_bos = seqs_cpu[b][k][1:] if cur_len > 1 else []
                    if len(seq_no_bos) + 1 < no_repeat_ngram_size:
                        continue
                    n = no_repeat_ngram_size
                    forb = {}
                    ln = len(seq_no_bos)
                    for i in range(ln - n + 1):
                        prefix = tuple(seq_no_bos[i:i + n - 1])
                        nxt = seq_no_bos[i + n - 1]
                        forb.setdefault(prefix, set()).add(nxt)
                    cur_prefix = tuple(seq_no_bos[-(n - 1):]) if n - 1 > 0 else tuple()
                    banned = forb.get(cur_prefix, None)
                    if banned:
                        idxs = torch.tensor(list(banned), dtype=torch.long, device=device)
                        logp[b, k, idxs] = -1e9

        # candidate scores
        total = scores.unsqueeze(-1) + logp  # (B, beam, V)
        flat = total.view(B, beam_size * V)  # (B, beam*V)
        topk_scores, topk_idx = torch.topk(flat, k=beam_size, dim=-1)  # (B, beam)
        prev_beam = topk_idx // V  # (B, beam)
        token_idx = topk_idx % V   # (B, beam)

        # выбрать предыдущие seqs согласно prev_beam (векторно)
        prev_beam_idx = prev_beam.unsqueeze(-1).expand(-1, -1, cur_len)  # (B, beam, cur_len)
        selected_prev = torch.gather(seqs, 1, prev_beam_idx)  # (B, beam, cur_len)

        # формируем новые seqs и scores
        new_seqs = torch.cat([selected_prev, token_idx.unsqueeze(-1)], dim=2)  # (B, beam, cur_len+1)
        new_scores = topk_scores  # (B, beam)
        prev_finished = torch.gather(is_finished, 1, prev_beam)  # (B, beam)
        new_finished = prev_finished | (token_idx == EOS)

        # Обработка EOS кандидатов: векторно обновляем best_completed_score и best_completed_seq
        if (token_idx == EOS).any():
            eos_mask = (token_idx == EOS)  # (B, beam) bool
            # длина вывода without BOS для нормализации = cur_len (см. вывод ниже)
            length_for_norm = cur_len  # integer scalar
            denom = _length_norm(length_for_norm, length_penalty)
            # candidate normalized score (для EOS только)
            cand_norm = torch.full((B, beam_size), float("-1e9"), device=device)
            cand_norm = torch.where(eos_mask, new_scores / denom, cand_norm)  # (B, beam)
            # получить лучший кандидат per example
            best_cand_vals, best_cand_idx = cand_norm.max(dim=1)  # (B,), (B,)
            # где лучше чем текущее best_completed_score — обновляем (векторно)
            better = best_cand_vals > best_completed_score
            if better.any():
                # выбираем seqs corresponding to best_cand_idx (векторно)
                # new_seqs_no_bos: (B, beam, cur_len+1 -1) = (B, beam, cur_len)
                new_seqs_no_bos = new_seqs[:, :, 1:]  # drop BOS, shape (B,beam,cur_len)
                sel_idx = best_cand_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, cur_len)  # (B,1,cur_len)
                picked = torch.gather(new_seqs_no_bos, 1, sel_idx).squeeze(1)  # (B, cur_len)
                # обновляем best_completed_seq и best_completed_score для тех, где better True
                best_completed_seq[better, :cur_len] = picked[better, :cur_len]
                best_completed_score[better] = best_cand_vals[better]

        # назначение переменных на следующий шаг
        seqs = new_seqs
        scores = new_scores
        is_finished = new_finished
        cur_len += 1

        # ранняя остановка (быстрая эвристика)
        if early_stopping:
            all_stop = True
            for b in range(B):
                if best_completed_score[b] <= -1e8:
                    all_stop = False
                    break
                # worst completed norm (мы храним только лучшую, поэтому используем её)
                # best_completed_score[b] уже нормализованная
                # best live possible:
                best_live_norm = float(scores[b].max().item()) / _length_norm(cur_len - 1, length_penalty)
                if best_live_norm > best_completed_score[b].item():
                    all_stop = False
                    break
            if all_stop:
                break

        if (~is_finished).sum().item() == 0:
            break

    # finalize: для каждого примера выбираем best_completed_seq, иначе лучшую живую гипотезу
    results = []
    # переносим необходимые тензоры на CPU один раз
    best_seq_cpu = best_completed_seq.cpu().numpy()
    seqs_cpu = seqs.cpu().numpy()
    scores_cpu = scores.cpu().numpy()
    for b in range(B):
        if best_completed_score[b] > -1e8:
            seq_tokens = best_seq_cpu[b].tolist()
            # удалить trailing PAD и EOS если есть (PAD id likely >= vocab_size and will be filtered later)
            # оставить только реальные токены (<vocab_size)
            results.append(seq_tokens)
        else:
            # choose best live
            best_idx = int(scores_cpu[b].argmax())
            seq_tokens = seqs_cpu[b, best_idx].tolist()
            # drop BOS
            if len(seq_tokens) > 0 and seq_tokens[0] == BOS:
                seq_tokens = seq_tokens[1:]
            # if last is EOS, drop it
            if len(seq_tokens) > 0 and seq_tokens[-1] == EOS:
                seq_tokens = seq_tokens[:-1]
            results.append(seq_tokens)

    return results


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
    Перевод набора строк. mode: "beam" или "greedy".
    Возвращает список строк и записывает файл.
    """
    assert mode in ("beam", "greedy")
    model.eval()
    PAD = int(dataset.PAD)
    preds = []

    for i in range(0, len(input_lines), batch_size):
        batch_lines = input_lines[i:i+batch_size]
        encoded = [dataset.encode(x) for x in batch_lines]
        max_src = max(len(s) for s in encoded)
        src_tensor = torch.full((len(encoded), max_src), PAD, dtype=torch.long)
        for j, seq in enumerate(encoded):
            src_tensor[j, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        src_tensor = src_tensor.to(device)

        if mode == "greedy":
            ids = greedy_decode(model, src_tensor, dataset, max_decoding_len, device)
        else:
            ids = batched_beam_search_optimized(
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

        for seq in ids:
            # фильтруем токены, оставляем только реальные id < vocab_size (PAD/BOS/EOS часто за пределом vocab_size)
            filtered = [int(x) for x in seq if int(x) < dataset.vocab_size]
            text = dataset.sp.decode_ids(filtered) if filtered else ""
            preds.append(text)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in preds:
            f.write(line.strip() + "\n")

    return preds