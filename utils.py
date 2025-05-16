import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import wandb, os

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

def train_model(model, train_loader, val_loader, optimizer, criterion, target_vocab,
                device, scheduler=None, num_epochs=10, teacher_forcing_ratio=None, patience=5,
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
            train_token_acc = train_token_correct / train_token_total * 100
            train_acc = train_word_correct / train_word_total * 100

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
                    val_correct += correct
                    val_total += total
                elif accuracy_mode == 'word': 
                    preds_seq = outputs[:, 1:].argmax(2)
                    tgt_seq = tgt_batch[:, 1:]
                    correct, total = compute_sequence_accuracy(preds_seq, tgt_seq, pad_idx, min_match_ratio)
                    val_correct += correct
                    val_total += total
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
            val_token_acc = val_token_correct / val_token_total * 100
            val_acc = val_word_correct / val_word_total * 100

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
              f"Train Loss: {train_loss:.4f}, Acc (token): {train_token_acc:.2f}%, Acc (word): {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc (token): {val_token_acc:.2f}%, Acc (word): {val_acc:.2f}% ")
            
        # Step the learning rate scheduler if provided
        if scheduler is not None:
            scheduler.step(val_loss)
            print(f"Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")

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

def beam_search_decode(model, src_tensor, src_vocab, tgt_vocab,
                       beam_width:int=3, max_len=50, device='cpu'):
    """
    Beam search decoding for a trained Seq2Seq model.

    Args:
        model: Trained Seq2Seq model
        src_tensor: tensor of token indices (1D tensor) for the source sentence
        src_vocab: source vocabulary (to access <sos> etc.)
        tgt_vocab: target vocabulary (to access <sos>, <eos>)
        beam_width: number of beams to maintain
        max_len: maximum output length
        device: torch.device

    Returns:
        Best output sequence (list of token indices)
    """
    model.eval()

    with torch.no_grad():
        # --- Step 1: Encode source ---
        src_tensor = src_tensor.to(device)  # (1, src_len)
        encoder_outputs, hidden = model.encoder(src_tensor)

        # Beam = list of tuples: (sequence, score, hidden_state)
        beams = [([tgt_vocab.sos_idx], 0.0, hidden)]  # Start with SOS token

        completed_sequences = []

        for _ in range(max_len):
            new_beams = []
            for seq, score, hidden_state in beams:
                input_token = torch.tensor([seq[-1]], device=device)
                if seq[-1] == tgt_vocab.eos_idx:
                    completed_sequences.append((seq, score))
                    continue

                # Decoder step
                output, hidden_next = model.decoder(input_token, hidden_state)
                output = output.squeeze(1)  # (1, vocab_size)

                log_probs = torch.log_softmax(output, dim=-1)  # (1, vocab_size)
                topk_log_probs, topk_indices = log_probs.topk(beam_width)  # (1, beam)

                for i in range(beam_width):
                    token = topk_indices[0, i].item()
                    token_log_prob = topk_log_probs[0, i].item()

                    new_seq = seq + [token]
                    new_score = score + token_log_prob
                    new_beams.append((new_seq, new_score, hidden_next))

            # Keep top k beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # Stop early if all beams are done
            if all(seq[-1] == tgt_vocab.eos_idx for seq, _, _ in beams):
                completed_sequences.extend(beams)
                break

        # Add unfinished beams
        completed_sequences.extend([b for b in beams if b[0][-1] != tgt_vocab.eos_idx])

        # Sort all complete sequences by score and return best one
        completed_sequences = sorted(completed_sequences, key=lambda x: x[1], reverse=True)

        best_seq = completed_sequences[0][0]
        return best_seq

def test_model(model, test_loader, source_vocab, target_vocab, device, beam_width: int = 3, output_dir="predictions_vanilla"):
    os.makedirs(output_dir, exist_ok=True)
    
    token_correct = 0
    token_total = 0
    word_correct = 0
    word_total = 0

    predictions = []
    sources = []
    references = []

    with torch.no_grad():
        for i, (src_batch, tgt_batch) in enumerate(tqdm(test_loader, desc="Testing")):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            for j in range(src_batch.size(0)):
                src = src_batch[j].unsqueeze(0)  # shape: [1, src_len]
                tgt = tgt_batch[j].unsqueeze(0)  # shape: [1, tgt_len]

                # Beam search decoding
                pred_tokens = beam_search_decode(model, src_tensor=src, src_vocab=source_vocab, tgt_vocab=target_vocab, beam_width=beam_width, max_len=50, device=device)
                # Reference tokens (excluding SOS, including EOS)
                reference = tgt.squeeze(0).tolist()
                pred_tokens = pred_tokens + [target_vocab.pad_idx] * (len(reference) - len(pred_tokens))
        
                if target_vocab.eos_idx in pred_tokens:
                    pred_eos_idx = pred_tokens.index(target_vocab.eos_idx) + 1
                else:
                    len(pred_tokens)
                reference_eos_idx = reference.index(target_vocab.eos_idx) + 1
                list_len = max(pred_eos_idx, reference_eos_idx)
                pred_tokens = pred_tokens[1:list_len]
                reference = reference[1:list_len]
                # if target_vocab.eos_idx in pred_tokens:
                #     pred_tokens = pred_tokens[1:pred_tokens.index(target_vocab.eos_idx)+1]
                # else:
                #     pred_tokens = pred_tokens[1:]
                # if target_vocab.eos_idx in reference:
                #     reference = reference[1:reference.index(target_vocab.eos_idx)+1]
                # else:
                #     reference = reference[1:]

                if not torch.is_tensor(reference):
                    targets = torch.tensor(reference).to(device)
                else:
                    targets = reference
                if not torch.is_tensor(pred_tokens):
                    preds = torch.tensor(pred_tokens).to(device)
                else:
                    preds = pred_tokens

                correct, total = compute_token_accuracy(preds, targets, target_vocab.pad_idx)
                token_correct += correct
                token_total += total
                
                if pred_tokens == reference:
                    word_correct += 1
                word_total += 1
                print(word_total)
                # Save input/prediction/reference for logging
                src_tokens = src.squeeze(0).tolist()
                sources.append(source_vocab.decode(src_tokens))
                predictions.append(target_vocab.decode(pred_tokens))
                references.append(target_vocab.decode(reference))

    # Save predictions to file
    with open(os.path.join(output_dir, "predictions.txt"), "w") as f:
        for src, ref, pred in zip(sources, references, predictions):
            f.write(f"SOURCE     : {src}\n")
            f.write(f"REFERENCE  : {ref}\n")
            f.write(f"PREDICTION : {pred}\n")
            f.write(f"{'-'*50}\n")

    # Print accuracy
    token_acc = token_correct / token_total * 100
    word_acc = word_correct / word_total * 100

    print(f"Test Results:")
    print(f"Token Accuracy: {token_acc:.2f}%")
    print(f"Sequence Accuracy (Exact Match): {word_acc:.2f}%")

    # Optional: print a creative grid
    print("Sample Predictions:")
    for i in range(5):
        print(f"Input      : {sources[i]}")
        print(f"Reference  : {references[i]}")
        print(f"Prediction : {predictions[i]}")
        print(f"{'-'*60}")