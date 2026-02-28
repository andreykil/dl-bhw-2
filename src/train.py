# src/train.py

import torch
from typing import Callable, Optional, List
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu
import re
from pathlib import Path

# import inference functions (use existing implementation)
from src.inference import translate_file

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 14})


def detokenize_english(text: str) -> str:
    """
    Простая детокенизация для английского (убирает пробелы перед знаками,
    склеивает апострофы и сокращает множественные пробелы).
    Подходит для sacrebleu --tokenize none.
    """
    s = text.strip()
    s = re.sub(r'\s+([?!\.,;:%\)\]\}])', r'\1', s)
    s = re.sub(r'([\(\[\{„“"\'`])\s+', r'\1', s)
    s = re.sub(r'\s+([\)\]\}\"\’\'”])', r'\1', s)
    s = re.sub(r"\b([A-Za-z]+)\s+'\s+([A-Za-z]+)\b", r"\1'\2", s)
    s = re.sub(r"\s+'\s+s\b", r"'s", s)
    s = re.sub(r'(\.\s*){2,}', '...', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s


def _plot_metrics(train_losses: List[float],
                  val_losses: List[float],
                  train_ppls: List[float],
                  val_ppls: List[float],
                  val_bleus: Optional[List[float]] = None):
    clear_output(wait=True)

    ncols = 3 if val_bleus is not None else 2
    fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

    if ncols == 2:
        axs = [axs[0], axs[1]]
    else:
        axs = list(axs)

    # Loss
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].legend()
    axs[0].grid(True)

    # Perplexity
    axs[1].plot(range(1, len(train_ppls) + 1), train_ppls, label='train')
    axs[1].plot(range(1, len(val_ppls) + 1), val_ppls, label='val')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('perplexity')
    axs[1].legend()
    axs[1].grid(True)

    # BLEU (val)
    if val_bleus is not None:
        axs[2].plot(range(1, len(val_bleus) + 1), val_bleus, label='val-bleu')
        axs[2].set_xlabel('epoch (bleu points)')
        axs[2].set_ylabel('BLEU')
        axs[2].legend()
        axs[2].grid(True)

    plt.tight_layout()
    plt.show()


def training_epoch(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   criterion,
                   train_loader,
                   pad_idx: int,
                   teacher_forcing_ratio: float = 0.5,
                   tqdm_desc: Optional[str] = None):
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    loader = tqdm(train_loader, desc=tqdm_desc) if tqdm_desc else train_loader
    for src, src_lens, tgt, tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)

        optimizer.zero_grad()
        outputs = model(src, src_lens, tgt, pad_idx, teacher_forcing_ratio=teacher_forcing_ratio)
        # outputs shape: (batch, tgt_len-1, vocab)
        target = tgt[:, 1:]  # skip BOS
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    return avg_loss, ppl


@torch.no_grad()
def validation_epoch(model: torch.nn.Module,
                     criterion,
                     val_loader,
                     pad_idx: int,
                     teacher_forcing_ratio: float = 0.0,
                     tqdm_desc: Optional[str] = None):
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    loader = tqdm(val_loader, desc=tqdm_desc) if tqdm_desc else val_loader
    for src, src_lens, tgt, tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)

        outputs = model(src, src_lens, tgt, pad_idx, teacher_forcing_ratio=teacher_forcing_ratio)
        target = tgt[:, 1:]
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    return avg_loss, ppl


def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler,
          criterion,
          train_loader,
          val_loader,
          pad_idx: int,
          num_epochs: int,
          *,
          val_dataset=None,                     # TranslationDataset instance (optional, for BLEU)
          max_decoding_len: int = 50,
          detokenize_fn: Optional[Callable[[str], str]] = detokenize_english,
          bleu_every: int = 1,                 # compute BLEU every N epochs (1 => every epoch)
          tmp_val_out: str = "outputs/val_predictions.en",
          device: torch.device = torch.device('cpu'),
          plot: bool = True):

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_ppls: List[float] = []
    val_ppls: List[float] = []
    val_bleus: Optional[List[float]] = [] if val_dataset is not None else None

    for epoch in range(1, num_epochs + 1):
        train_loss, train_ppl = training_epoch(
            model, optimizer, criterion, train_loader, pad_idx,
            teacher_forcing_ratio=0.5, tqdm_desc=f"Train {epoch}/{num_epochs}"
        )

        val_loss, val_ppl = validation_epoch(
            model, criterion, val_loader, pad_idx,
            teacher_forcing_ratio=0.0, tqdm_desc=f"Val {epoch}/{num_epochs}"
        )

        # update scheduler: if ReduceLROnPlateau requires metric -> pass val_loss
        if scheduler is not None:
            try:
                scheduler.step(val_loss)
            except TypeError:
                scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        bleu_score = None
        # compute BLEU every bleu_every epochs (and only if val_dataset provided)
        if (val_dataset is not None) and (epoch % bleu_every == 0):
            val_input_lines = list(val_dataset.src_lines)
            Path(tmp_val_out).parent.mkdir(parents=True, exist_ok=True)
            translate_file(model=model,
                        dataset=val_dataset,
                        input_lines=val_input_lines,
                        max_decoding_len=max_decoding_len,
                        device=device,
                        output_path=tmp_val_out)
            preds = Path(tmp_val_out).read_text(encoding="utf-8").splitlines()
            refs_raw = list(val_dataset.tgt_lines)
            # detokenize preds and refs
            if detokenize_fn is not None:
                preds_detok = [detokenize_fn(s) for s in preds]
                refs_detok = [detokenize_fn(s) for s in refs_raw]
                bleu = sacrebleu.corpus_bleu(preds_detok, [refs_detok], tokenize='none')
            else:
                bleu = sacrebleu.corpus_bleu(preds, [refs_raw])
            bleu_score = float(bleu.score)
            # append into val_bleus aligned with epoch index:
            if val_bleus is not None:
                val_bleus.append(bleu_score)

            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f} val_BLEU={bleu_score:.2f}")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

        if plot:
            _plot_metrics(train_losses, val_losses, train_ppls, val_ppls, val_bleus)

    return train_losses, val_losses, train_ppls, val_ppls, val_bleus