# src/train.py
import math
from pathlib import Path
from typing import Optional, List

import torch
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu
import numpy as np

from src.inference import translate_file

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 14})


class InverseSqrtScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int = 512, warmup_steps: int = 4000, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = max(1, int(warmup_steps))
        self.factor = float(factor)
        self._step = 0

    def _lr_scale(self, step: int) -> float:
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def step(self):
        self._step += 1
        lr = self.factor * self._lr_scale(self._step)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_lr(self) -> float:
        if self._step == 0:
            return 0.0
        return self.factor * self._lr_scale(self._step)


# ------------------------------------------------
# plotting
# ------------------------------------------------

def _plot_metrics(train_losses: List[float],
                  val_losses: List[float],
                  train_ppls: List[float],
                  val_ppls: List[float],
                  bleu_greedy: Optional[List[float]] = None,
                  bleu_beam: Optional[List[float]] = None):
    """
    Рисует 2 графика (loss, ppl) и, при наличии BLEU, третий с двумя кривыми BLEU.
    BLEU списки могут содержать math.nan — точки в этих местах не рисуются.
    """
    clear_output(wait=True)

    ncols = 3 if bleu_greedy is not None else 2
    fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

    if ncols == 2:
        axs = [axs[0], axs[1]]
    else:
        axs = list(axs)

    # Loss
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_xlabel('epoch'); axs[0].set_ylabel('loss'); axs[0].legend(); axs[0].grid(True)

    # Perplexity
    axs[1].plot(range(1, len(train_ppls) + 1), train_ppls, label='train')
    axs[1].plot(range(1, len(val_ppls) + 1), val_ppls, label='val')
    axs[1].set_xlabel('epoch'); axs[1].set_ylabel('perplexity'); axs[1].legend(); axs[1].grid(True)

    # BLEU (greedy + beam)
    if bleu_greedy is not None and bleu_beam is not None:
        epochs = np.arange(1, len(bleu_greedy) + 1)
        # convert to numpy arrays with nan for missing
        g = np.array(bleu_greedy, dtype=float)
        b = np.array(bleu_beam, dtype=float)
        # plot points where finite
        axs[2].plot(epochs[np.isfinite(g)], g[np.isfinite(g)], label='BLEU greedy', marker='o')
        axs[2].plot(epochs[np.isfinite(b)], b[np.isfinite(b)], label='BLEU beam', marker='o')
        axs[2].set_xlabel('epoch'); axs[2].set_ylabel('BLEU'); axs[2].legend(); axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------
# training / validation loops
# ------------------------------------------------

def training_epoch(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[InverseSqrtScheduler],
                   criterion,
                   train_loader,
                   pad_idx: int,
                   device: torch.device,
                   max_grad_norm: float = 1.0,
                   gradient_accumulation_steps: int = 1,
                   word_dropout: float = 0.0,
                   tqdm_desc: Optional[str] = None):
    """
    Один проход по train_loader с поддержкой gradient accumulation и простого word dropout.
    word_dropout: дробь (0..1) — вероятность подмены входного токена на PAD (не трогает BOS/EOS/PAD).
    """

    model.train()
    total_loss = 0.0
    total_tokens = 0

    loader = tqdm(train_loader, desc=tqdm_desc) if tqdm_desc else train_loader

    optimizer.zero_grad()
    for step, (src, src_lens, tgt, tgt_lens) in enumerate(loader):
        src = src.to(device)
        tgt = tgt.to(device)

        # word dropout augmentation (only for src)
        if word_dropout > 0.0 and word_dropout < 1.0:
            # expect dataset available via train_loader.dataset
            ds = train_loader.dataset
            BOS, EOS, PAD = getattr(ds, 'BOS'), getattr(ds, 'EOS'), getattr(ds, 'PAD')
            # generate mask
            drop_mask = (torch.rand(src.shape, device=src.device) < word_dropout)
            protect = (src == PAD) | (src == BOS) | (src == EOS)
            drop_mask = drop_mask & (~protect)
            src_input = src.masked_fill(drop_mask, PAD)
        else:
            src_input = src

        # forward
        outputs = model(src_input, tgt)  # (batch, tgt_len-1, vocab)
        target = tgt[:, 1:]

        raw_loss = criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))
        loss = raw_loss / float(gradient_accumulation_steps)
        loss.backward()

        # accumulate statistics on raw loss (per-token)
        total_loss += raw_loss.item() * target.numel()
        total_tokens += target.numel()

        # step optimizer when accumulated enough
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                # scheduler.step() only when optimizer.step() executed
                scheduler.step()
            optimizer.zero_grad()

    # handle leftover gradients (if dataset size not divisible by accumulation)
    # (if last step wasn't a multiple, gradients already applied above)
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, ppl


@torch.no_grad()
def validation_epoch(model: torch.nn.Module,
                     criterion,
                     val_loader,
                     pad_idx: int,
                     device: torch.device,
                     tqdm_desc: Optional[str] = None):
    """
    Returns avg_loss and perplexity on val_loader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    loader = tqdm(val_loader, desc=tqdm_desc) if tqdm_desc else val_loader

    for src, src_lens, tgt, tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        outputs = model(src, tgt)
        target = tgt[:, 1:]
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return avg_loss, ppl


# ------------------------------------------------
# full training
# ------------------------------------------------

def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: Optional[InverseSqrtScheduler],
          criterion,
          train_loader,
          val_loader,
          pad_idx: int,
          num_epochs: int,
          *,
          val_dataset=None,
          max_decoding_len: int = 50,
          greedy_bleu_every: int = 1,
          beam_bleu_every: int = 1,
          bleu_best_model: str = 'greedy',  # 'greedy' or 'beam'
          tmp_val_out: str = "outputs/val_predictions.en",
          device: torch.device = torch.device('cpu'),
          inference_batch_size: int = 64,
          no_repeat_ngram_size: int = 0,
          gradient_accumulation_steps: int = 1,
          word_dropout: float = 0.0,
          plot: bool = True):
    """
    Полный тренинг:
      - gradient accumulation
      - word dropout (регуляризация)
      - BLEU (greedy и beam) с разными частотами
      - checkpointing: best_by_loss и best_by_bleu (greedy|beam)
    """

    # bookkeeping
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_ppls: List[float] = []
    val_ppls: List[float] = []

    # BLEU histories: store math.nan for epochs where the corresponding BLEU wasn't computed
    bleu_greedy: List[float] = []
    bleu_beam: List[float] = []

    # best trackers
    best_val_loss = float('inf')
    best_loss_path = Path("checkpoints/best_by_loss.pt")
    best_bleu = float('-inf')
    best_bleu_path = Path(f"checkpoints/best_by_bleu_{bleu_best_model}.pt")

    # create checkpoints dir
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path(tmp_val_out).parent.mkdir(parents=True, exist_ok=True)

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        # ------ train epoch ------
        train_loss, train_ppl = training_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_loader=train_loader,
            pad_idx=pad_idx,
            device=device,
            max_grad_norm=1.0,
            gradient_accumulation_steps=gradient_accumulation_steps,
            word_dropout=word_dropout,
            tqdm_desc=f"Train {epoch}/{num_epochs}"
        )

        # ------ validation loss ------
        val_loss, val_ppl = validation_epoch(
            model=model,
            criterion=criterion,
            val_loader=val_loader,
            pad_idx=pad_idx,
            device=device,
            tqdm_desc=f"Val {epoch}/{num_epochs}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        # checkpoint by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, str(best_loss_path))

        # ----- BLEU computations -----
        # we compute greedy BLEU if epoch % greedy_bleu_every == 0 else append nan
        greedy_bleu_val = math.nan
        beam_bleu_val = math.nan

        if val_dataset is not None:
            refs = list(val_dataset.tgt_lines)
            val_input_lines = list(val_dataset.src_lines)

            # GREEDY
            if greedy_bleu_every > 0 and (epoch % greedy_bleu_every == 0):
                preds_greedy = translate_file(
                    model=model,
                    dataset=val_dataset,
                    input_lines=val_input_lines,
                    max_decoding_len=max_decoding_len,
                    device=device,
                    output_path=tmp_val_out,
                    batch_size=inference_batch_size,
                    mode="greedy"
                )
                greedy_bleu_val = sacrebleu.corpus_bleu(preds_greedy, [refs]).score

            # BEAM
            if beam_bleu_every > 0 and (epoch % beam_bleu_every == 0):
                preds_beam = translate_file(
                    model=model,
                    dataset=val_dataset,
                    input_lines=val_input_lines,
                    max_decoding_len=max_decoding_len,
                    device=device,
                    output_path=tmp_val_out,
                    batch_size=inference_batch_size,
                    mode="beam",
                    beam_size=5,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
                beam_bleu_val = sacrebleu.corpus_bleu(preds_beam, [refs]).score

            # decide whether to save best_by_bleu
            # choose relevant metric for comparison depending on bleu_best_model param
            compare_bleu = None
            if bleu_best_model == 'greedy':
                compare_bleu = greedy_bleu_val
            else:
                compare_bleu = beam_bleu_val

            if not (compare_bleu is None or (isinstance(compare_bleu, float) and math.isnan(compare_bleu))):
                # compare with stored best
                if compare_bleu > best_bleu:
                    best_bleu = compare_bleu
                    # save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'bleu': compare_bleu,
                        'bleu_mode': bleu_best_model
                    }, str(best_bleu_path))

        # append BLEU history lists (math.nan where not computed)
        bleu_greedy.append(greedy_bleu_val)
        bleu_beam.append(beam_bleu_val)

        # print status
        if val_dataset is not None:
            gstr = f"{greedy_bleu_val:.2f}" if not math.isnan(greedy_bleu_val) else " - "
            bstr = f"{beam_bleu_val:.2f}" if not math.isnan(beam_bleu_val) else " - "
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.2f} BLEU_greedy={gstr} BLEU_beam={bstr}"
            )
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

        # plot
        if plot:
            _plot_metrics(train_losses, val_losses, train_ppls, val_ppls, bleu_greedy, bleu_beam)

    # final return
    return train_losses, val_losses, train_ppls, val_ppls, bleu_greedy, bleu_beam