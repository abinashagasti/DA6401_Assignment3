import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import torch

def train_model(model, train_loader, val_loader, optimizer, criterion,
                target_vocab, device, num_epochs=10, teacher_forcing_ratio=1.0,
                patience=3, min_delta=0.001):
    """
    Trains a Seq2Seq model.

    Parameters:
        model: Seq2Seq model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        optimizer: optimizer
        criterion: loss function
        target_vocab: target Vocab object to access <pad> token index
        device: torch.device (CPU, CUDA, MPS)
        num_epochs: maximum number of epochs
        teacher_forcing_ratio: ratio of teacher forcing to use
        patience: how many epochs to wait for improvement before stopping
        min_delta: minimum change in validation loss to qualify as improvement
    """

    pad_idx = target_vocab.pad_idx
    best_val_loss = float('inf')
    no_improvement = 0

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for src_batch, tgt_batch in train_bar:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()
            outputs = model(src_batch, tgt_batch, teacher_forcing_ratio=teacher_forcing_ratio)

            output_dim = outputs.size(-1)
            outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
            tgt_flat = tgt_batch[:, 1:].reshape(-1)

            loss = criterion(outputs_flat, tgt_flat)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = outputs_flat.argmax(1)
            mask = tgt_flat != pad_idx
            train_correct += (preds == tgt_flat).masked_select(mask).sum().item()
            train_total += mask.sum().item()

            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total * 100

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)

        with torch.no_grad():
            for src_batch, tgt_batch in val_bar:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)

                outputs = model(src_batch, tgt_batch, teacher_forcing_ratio=0.0)

                outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
                tgt_flat = tgt_batch[:, 1:].reshape(-1)

                loss = criterion(outputs_flat, tgt_flat)
                val_loss += loss.item()

                preds = outputs_flat.argmax(1)
                mask = tgt_flat != pad_idx
                val_correct += (preds == tgt_flat).masked_select(mask).sum().item()
                val_total += mask.sum().item()

                val_bar.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total * 100

        # --- Logging ---
        print(f"Epoch {epoch:02d} ➤ "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # --- Early Stopping Check ---
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improvement = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }
            torch.save(checkpoint,'checkpoint.pth')
            print(f"Saved checkpoint at epoch {epoch} with val_loss {val_loss:.4f}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
