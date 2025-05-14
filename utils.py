import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import wandb

def compute_token_accuracy(preds, targets, pad_idx):
    mask = targets != pad_idx
    correct = (preds == targets).masked_select(mask).sum().item()
    total = mask.sum().item()
    return correct, total

def compute_sequence_accuracy(preds, targets, pad_idx, min_match_ratio=1.0):
    """
    Returns number of sequences where predicted sequence matches target,
    optionally allowing partial matches (e.g. 90% tokens match).

    Inputs:
        preds: predictions made for given batch
        targets: target outputs for given batch
        pad_idx: padding token index
        min_match_ratio: fraction of character match to allow for correct word translation
    """
    correct = 0
    total = preds.size(0)

    for pred_seq, tgt_seq in zip(preds, targets):
        pred_seq = pred_seq.tolist()
        tgt_seq = tgt_seq.tolist()

        pred_trimmed = [x for x in pred_seq if x != pad_idx]
        tgt_trimmed = [x for x in tgt_seq if x != pad_idx]

        match_count = sum(p == t for p, t in zip(pred_trimmed, tgt_trimmed))
        required = int(min(len(tgt_trimmed), len(pred_trimmed)) * min_match_ratio)

        if match_count >= required:
            correct += 1

    return correct, total

# def compute_sequence_accuracy(preds, targets, pad_idx):
#     # preds: (batch_size, seq_len), targets: (batch_size, seq_len)
#     batch_size = preds.size(0)
#     correct = 0
#     total = batch_size
#     for i in range(batch_size):
#         pred_seq = preds[i].tolist()
#         tgt_seq = targets[i].tolist()

#         # Remove padding and truncate at EOS if needed
#         pred_trimmed = [x for x in pred_seq if x != pad_idx]
#         tgt_trimmed = [x for x in tgt_seq if x != pad_idx]

#         if pred_trimmed == tgt_trimmed:
#             correct += 1
#     return correct, total

def train_model(model, train_loader, val_loader, optimizer, criterion, target_vocab,
                device, num_epochs=10, teacher_forcing_ratio=None, patience=3,
                min_delta=0.001, accuracy_mode='token', min_match_ratio=1.0, wandb_log=False):
    """
    Trains a Seq2Seq model.

    Inputs:
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
        accuracy_mode: method to evaluate accuracy during training and validation, choices = ['token', 'word', 'both']
        min_match_ratio: fraction of character match to allow for correct word translation
    """

    pad_idx = target_vocab.pad_idx
    best_val_loss = float('inf')
    no_improvement = 0

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_token_correct = 0
        train_word_correct = 0
        train_total = 0
        train_token_total = 0
        train_word_total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for src_batch, tgt_batch in train_bar:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            optimizer.zero_grad()

            if teacher_forcing_ratio is None:
                teacher_forcing_ratio = max(0.5 * (0.99 ** epoch), 0.1)

            outputs = model(src_batch, tgt_batch, teacher_forcing_ratio=teacher_forcing_ratio)

            output_dim = outputs.size(-1)
            outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
            tgt_flat = tgt_batch[:, 1:].reshape(-1)

            loss = criterion(outputs_flat, tgt_flat)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = outputs_flat.argmax(1)
            if accuracy_mode == 'token':
                correct, total = compute_token_accuracy(preds, tgt_flat, pad_idx)
                train_correct += correct
                train_total += total
            elif accuracy_mode == 'word': 
                preds_seq = outputs[:, 1:].argmax(2)
                tgt_seq = tgt_batch[:, 1:]
                correct, total = compute_sequence_accuracy(preds_seq, tgt_seq, pad_idx, min_match_ratio)
                train_correct += correct
                train_total += total
            else: 
                correct, total = compute_token_accuracy(preds, tgt_flat, pad_idx)
                train_token_correct += correct
                train_token_total += total
                preds_seq = outputs[:, 1:].argmax(2)
                tgt_seq = tgt_batch[:, 1:]
                correct, total = compute_sequence_accuracy(preds_seq, tgt_seq, pad_idx, min_match_ratio)
                train_word_correct += correct
                train_word_total += total
           
            # mask = tgt_flat != pad_idx
            # train_correct += (preds == tgt_flat).masked_select(mask).sum().item()
            # train_total += mask.sum().item()

            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        if accuracy_mode == 'token' or accuracy_mode == 'word':
            train_acc = train_correct / train_total * 100
        else:
            train_acc = train_token_correct / train_token_total * 100
            train_word_acc = train_word_correct / train_word_total * 100

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_token_correct = 0
        val_word_correct = 0
        val_total = 0
        val_token_total = 0
        val_word_total = 0

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
                if accuracy_mode == 'token':
                    correct, total = compute_token_accuracy(preds, tgt_flat, pad_idx)
                    train_correct += correct
                    train_total += total
                elif accuracy_mode == 'word': 
                    preds_seq = outputs[:, 1:].argmax(2)
                    tgt_seq = tgt_batch[:, 1:]
                    correct, total = compute_sequence_accuracy(preds_seq, tgt_seq, pad_idx, min_match_ratio)
                    train_correct += correct
                    train_total += total
                else: 
                    correct, total = compute_token_accuracy(preds, tgt_flat, pad_idx)
                    val_token_correct += correct
                    val_token_total += total
                    preds_seq = outputs[:, 1:].argmax(2)
                    tgt_seq = tgt_batch[:, 1:]
                    correct, total = compute_sequence_accuracy(preds_seq, tgt_seq, pad_idx, min_match_ratio)
                    val_word_correct += correct
                    val_word_total += total

                # mask = tgt_flat != pad_idx
                # val_correct += (preds == tgt_flat).masked_select(mask).sum().item()
                # val_total += mask.sum().item()

                val_bar.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        if accuracy_mode == 'token' or accuracy_mode == 'word':
            val_acc = val_correct / val_total * 100
        else:
            val_acc = val_token_correct / val_token_total * 100
            val_word_acc = val_word_correct / val_word_total * 100

        # --- Logging ---
        # Log to wandb if enabled
        if wandb_log:
                wandb.log({
                    "epoch": epoch,
                    "training_loss": train_loss,
                    "training_accuracy": train_acc,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc
                })    
        
        if accuracy_mode == 'token' or accuracy_mode == 'word':
            print(f"Epoch {epoch:02d} ➤ "
              f"Train Loss: {train_loss:.4f}, Acc ({accuracy_mode}): {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc ({accuracy_mode}): {val_acc:.2f}%")
        else:
            print(f"Epoch {epoch:02d} ➤ "
              f"Train Loss: {train_loss:.4f}, Acc (token): {train_acc:.2f}%, Acc (word): {train_word_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc (token): {val_acc:.2f}%, Acc (word): {val_word_acc:.2f}% ")

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
