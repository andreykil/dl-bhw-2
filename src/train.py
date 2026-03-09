import math
from pathlib import Path
from typing import Optional, List

import torch
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu

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

def _plot_metrics(train_losses,
                  val_losses,
                  train_ppls,
                  val_ppls,
                  bleu_greedy: Optional[List[float]] = None,
                  bleu_beam: Optional[List[float]] = None):

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

    # BLEU
    if bleu_greedy is not None:
        epochs = range(1, len(bleu_greedy) + 1)

        axs[2].plot(epochs, bleu_greedy, label='BLEU greedy', marker='o')
        # axs[2].plot(epochs, bleu_beam, label='BLEU beam', marker='o')

        axs[2].set_xlabel('epoch')
        axs[2].set_ylabel('BLEU')
        axs[2].legend()
        axs[2].grid(True)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------
# training / validation loops
# ------------------------------------------------

def training_epoch(model,
                   optimizer,
                   scheduler,
                   criterion,
                   train_loader,
                   pad_idx,
                   device,
                   max_grad_norm=1.0,
                   tqdm_desc=None):

    model.train()
    total_loss = 0.0
    total_tokens = 0

    loader = tqdm(train_loader, desc=tqdm_desc) if tqdm_desc else train_loader

    for src, src_lens, tgt, tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        outputs = model(src, tgt)
        target = tgt[:, 1:]

        loss = criterion(outputs.reshape(-1, outputs.size(-1)), target.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return avg_loss, ppl


@torch.no_grad()
def validation_epoch(model,
                     criterion,
                     val_loader,
                     pad_idx,
                     device,
                     tqdm_desc=None):

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

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    return avg_loss, ppl


# ------------------------------------------------
# full training
# ------------------------------------------------

def train(model,
          optimizer,
          scheduler,
          criterion,
          train_loader,
          val_loader,
          pad_idx,
          num_epochs,
          *,
          val_dataset=None,
          max_decoding_len=50,
          bleu_every=1,
          tmp_val_out="outputs/val_predictions.en",
          device=torch.device("cpu"),
          inference_batch_size=64,
          no_repeat_ngram_size=3,
          plot=True):

    train_losses = []
    val_losses = []

    train_ppls = []
    val_ppls = []

    bleu_greedy = []
    bleu_beam = []

    model.to(device)

    for epoch in range(1, num_epochs + 1):

        train_loss, train_ppl = training_epoch(
            model, optimizer, scheduler, criterion,
            train_loader, pad_idx, device,
            tqdm_desc=f"Train {epoch}/{num_epochs}"
        )

        val_loss, val_ppl = validation_epoch(
            model, criterion, val_loader, pad_idx, device,
            tqdm_desc=f"Val {epoch}/{num_epochs}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        # ------------------------------------------------
        # BLEU
        # ------------------------------------------------

        if (val_dataset is not None) and (epoch % bleu_every == 0):

            val_input_lines = list(val_dataset.src_lines)
            refs = list(val_dataset.tgt_lines)

            Path(tmp_val_out).parent.mkdir(parents=True, exist_ok=True)

            # GREEDY
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

            bleu1 = sacrebleu.corpus_bleu(preds_greedy, [refs]).score

            # BEAM
            # preds_beam = translate_file(
            #     model=model,
            #     dataset=val_dataset,
            #     input_lines=val_input_lines,
            #     max_decoding_len=max_decoding_len,
            #     device=device,
            #     output_path=tmp_val_out,
            #     batch_size=inference_batch_size,
            #     mode="beam",
            #     beam_size=5,
            #     no_repeat_ngram_size=no_repeat_ngram_size,
            # )

            # bleu2 = sacrebleu.corpus_bleu(preds_beam, [refs]).score

            bleu_greedy.append(bleu1)
            # bleu_beam.append(bleu2)

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.2f} "
                f"BLEU_greedy={bleu1:.2f} "
                # f"BLEU_beam={bleu2:.2f}"
            )

        else:

            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.2f}"
            )

        if plot:
            _plot_metrics(
                train_losses,
                val_losses,
                train_ppls,
                val_ppls,
                bleu_greedy,
                # bleu_beam
            )

    return train_losses, val_losses, train_ppls, val_ppls, bleu_greedy, bleu_beam