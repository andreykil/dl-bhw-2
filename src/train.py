# src/train.py

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import tqdm


sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 14})


def plot_losses(train_losses, val_losses, train_ppls, val_ppls):
    clear_output(wait=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    # Loss
    axs[0].plot(train_losses, label="train")
    axs[0].plot(val_losses, label="val")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")
    axs[0].legend()

    # Perplexity
    axs[1].plot(train_ppls, label="train")
    axs[1].plot(val_ppls, label="val")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("perplexity")
    axs[1].legend()

    plt.show()


def training_epoch(model, optimizer, criterion, train_loader, pad_idx, tqdm_desc=None):
    model.train()

    device = next(model.parameters()).device
    total_loss = 0
    total_tokens = 0

    loader = tqdm(train_loader, desc=tqdm_desc) if tqdm_desc else train_loader

    for src, src_lens, tgt, tgt_lens in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_lens = src_lens.to(device)

        optimizer.zero_grad()

        outputs = model(
            src,
            src_lens,
            tgt,
            pad_idx,
            teacher_forcing_ratio=0.5
        )  # (batch, tgt_len-1, vocab)

        target = tgt[:, 1:]

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            target.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


@torch.no_grad()
def validation_epoch(model, criterion, val_loader, pad_idx, tqdm_desc=None):
    model.eval()

    device = next(model.parameters()).device
    total_loss = 0
    total_tokens = 0

    loader = tqdm(val_loader, desc=tqdm_desc) if tqdm_desc else val_loader

    for src, src_lens, tgt, tgt_lens in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_lens = src_lens.to(device)

        outputs = model(
            src,
            src_lens,
            tgt,
            pad_idx,
            teacher_forcing_ratio=0.0
        )

        target = tgt[:, 1:]

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            target.reshape(-1)
        )

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def train(
    model,
    optimizer,
    scheduler,
    criterion,
    train_loader,
    val_loader,
    pad_idx,
    num_epochs,
    plot=True
):
    train_losses, val_losses = [], []
    train_ppls, val_ppls = [], []

    for epoch in range(1, num_epochs + 1):

        train_loss, train_ppl = training_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            pad_idx,
            tqdm_desc=f"Train {epoch}/{num_epochs}"
        )

        val_loss, val_ppl = validation_epoch(
            model,
            criterion,
            val_loader,
            pad_idx,
            tqdm_desc=f"Val {epoch}/{num_epochs}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        if plot:
            plot_losses(train_losses, val_losses, train_ppls, val_ppls)

    return train_losses, val_losses, train_ppls, val_ppls