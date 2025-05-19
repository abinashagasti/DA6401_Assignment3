import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import wandb, os, random
from io import BytesIO

import pandas as pd
import Levenshtein, html
from Levenshtein import distance as edit_distance
from tabulate import tabulate
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns

def visualisation_table(file_path="predictions_vanilla/predictions.txt", n = 10, wandb_log = False):
    # Load TSV predictions file
    df = pd.read_csv(file_path, sep="\t")

    # Sample rows
    sampled_df = df.sample(n=n).reset_index(drop=True)

    # Compute edit distance
    sampled_df["EditDistance"] = sampled_df.apply(
        lambda row: edit_distance(str(row["REFERENCE"]), str(row["PREDICTION"])), axis=1
    )

    # Add status (correct / close / wrong)
    def get_status(ref, pred):
        if ref == pred:
            return "Correct"
        elif edit_distance(ref, pred) <= 2:
            return "Close"
        else:
            return "Wrong"

    sampled_df["Status"] = sampled_df.apply(
        lambda row: get_status(str(row["REFERENCE"]), str(row["PREDICTION"])), axis=1
    )

    # Display on terminal
    from tabulate import tabulate
    print(tabulate(sampled_df[["SOURCE", "REFERENCE", "PREDICTION", "EditDistance", "Status"]],
                   headers="keys", tablefmt="fancy_grid"))

    # Log to wandb if needed
    if wandb_log:
        wandb_table = wandb.Table(columns=["SOURCE", "REFERENCE", "PREDICTION", "EditDistance", "Status"])
        for _, row in sampled_df.iterrows():
            wandb_table.add_data(
                str(row["SOURCE"]),
                str(row["REFERENCE"]),
                str(row["PREDICTION"]),
                row["EditDistance"],
                row["Status"]
            )
        wandb.log({"Sample Predictions": wandb_table})

def visualisation_table_color(file_path="predictions_vanilla/predictions.txt", n=10, wandb_log=False):
    # Load TSV predictions file
    df = pd.read_csv(file_path, sep="\t")

    # Sample rows
    sampled_df = df.sample(n=n).reset_index(drop=True)

    # Compute edit distance
    sampled_df["EditDistance"] = sampled_df.apply(
        lambda row: edit_distance(str(row["REFERENCE"]), str(row["PREDICTION"])), axis=1
    )

    # Start HTML table with UTF-8 declaration
    html_table = """
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            table { border-collapse: collapse; font-family: sans-serif; }
            th, td { padding: 6px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
    <table>
        <tr>
            <th>SOURCE</th>
            <th>REFERENCE</th>
            <th>PREDICTION</th>
            <th>EDIT DISTANCE</th>
        </tr>
    """

    for _, row in sampled_df.iterrows():
        ref = row["REFERENCE"]
        pred = row["PREDICTION"]
        src = row["SOURCE"]
        ed = row["EditDistance"]

        if ref == pred:
            color = "#c8e6c9"
        elif ed <= 2:
            color = "#fff9c4"
        else:
            color = "#ffcdd2"

        html_table += f"""
        <tr style="background-color:{color}">
            <td>{src}</td>
            <td>{ref}</td>
            <td>{pred}</td>
            <td style="text-align:center;">{ed}</td>
        </tr>
        """

    html_table += "</table></body></html>"

    # Optional: Display locally
    from IPython.core.display import display, HTML as IPHTML
    display(IPHTML(html_table))

    # Log to wandb
    if wandb_log:
        wandb.log({"Prediction Table": wandb.Html(html_table)})

def compare_attention_vs_vanilla_sampled(vanilla_file="predictions_vanilla/predictions.txt",
                                         attention_file="predictions_attention/predictions.txt",
                                         sample_n=20,
                                         wandb_log=False,
                                         wandb_table_name="Attention_Vs_Vanilla_Sample"):

    # Load both prediction files
    df_vanilla = pd.read_csv(vanilla_file, sep="\t")
    df_attention = pd.read_csv(attention_file, sep="\t")

    assert len(df_vanilla) == len(df_attention), "Mismatch in number of predictions"

    # Collect cases where attention is correct and vanilla is wrong
    rows = []
    for i in range(len(df_vanilla)):
        src = str(df_vanilla.loc[i, "SOURCE"])
        ref = str(df_vanilla.loc[i, "REFERENCE"])
        pred_vanilla = str(df_vanilla.loc[i, "PREDICTION"])
        pred_attention = str(df_attention.loc[i, "PREDICTION"])

        is_vanilla_correct = pred_vanilla == ref
        is_attention_correct = pred_attention == ref

        if not is_vanilla_correct and is_attention_correct:
            rows.append({
                "SOURCE": src,
                "REFERENCE": ref,
                "VANILLA": pred_vanilla,
                "ATTENTION": pred_attention,
                "VANILLA_ED": edit_distance(pred_vanilla, ref),
                "ATTENTION_ED": edit_distance(pred_attention, ref)
            })

    result_df = pd.DataFrame(rows)
    print(f"✅ Found {len(result_df)} cases where attention succeeded but vanilla failed.")

    if result_df.empty:
        print("⚠️ No qualifying rows to sample or log.")
        return None

    # Sample from the result
    sampled_df = result_df.sample(n=min(sample_n, len(result_df)), random_state=42).reset_index(drop=True)

    # Log to wandb if requested
    if wandb_log:
        table = wandb.Table(columns=["SOURCE", "REFERENCE", "VANILLA", "ATTENTION", "VANILLA_ED", "ATTENTION_ED"])
        counter = 0
        for _, row in sampled_df.iterrows():
            table.add_data(row["SOURCE"], row["REFERENCE"], row["VANILLA"], row["ATTENTION"],
                           row["VANILLA_ED"], row["ATTENTION_ED"])
            if counter>sample_n:
                break
        wandb.log({wandb_table_name: table})
        print(f"📤 Logged a random sample of {len(sampled_df)} cases to W&B table `{wandb_table_name}`")

    return sampled_df


def sample_from_testset(test_loader, num_samples=10):
    all_src = []
    all_tgt = []

    # First, collect all examples from the test set
    for src_batch, tgt_batch in test_loader:
        all_src.extend(src_batch)
        all_tgt.extend(tgt_batch)

    # Now randomly pick `num_samples` indices
    indices = random.sample(range(len(all_src)), num_samples)

    # Extract the sampled sequences
    sampled_src = [all_src[i] for i in indices]
    sampled_tgt = [all_tgt[i] for i in indices]

    return sampled_src, sampled_tgt


def plot_attention_heatmaps(model, test_loader):
    
    sampled_src, sampled_tgt = sample_from_testset(test_loader) # get 10 samples from the test set
    


def plot_attention_heatmaps(model, test_loader, source_vocab, target_vocab, device,
                             prediction_file="predictions_attention/predictions.txt", 
                             num_samples=10, wandb_log=False):
    # Load the predictions
    df = pd.read_csv(prediction_file, sep="\t")
    assert len(df) > 0, "Prediction file is empty."

    # Sample indices
    sample_indices = random.sample(range(len(df)), num_samples)
    sampled_df = df.iloc[sample_indices].reset_index(drop=True)

    def decode(tokens, vocab):
        return [vocab.idx_to_token[idx] for idx in tokens if idx not in {vocab.pad_idx, vocab.eos_idx, vocab.sos_idx}]

    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    axs = axs.flatten()
    model.eval()
    count = 0

    with torch.no_grad():
        for i in range(num_samples):
            src_text = sampled_df.loc[i, "SOURCE"]
            pred_text = sampled_df.loc[i, "PREDICTION"]

            try:
                src_tokens = [source_vocab.token_to_idx[c] for c in src_text]
            except KeyError:
                print(f"Skipping unknown chars in: {src_text}")
                continue

            src_tensor = torch.tensor(
                [source_vocab.sos_idx] + src_tokens + [source_vocab.eos_idx], device=device
            ).unsqueeze(0)

            # Predict with attention
            pred_tokens, attention_weights = beam_search_decode_with_attention(
                model, src_tensor, source_vocab, target_vocab,
                beam_width=3, max_len=50, device=device
            )

            src_decoded = decode(src_tensor.squeeze().tolist(), source_vocab)
            pred_decoded = decode(pred_tokens, target_vocab)

            if attention_weights is None or attention_weights.shape[0] == 0:
                print(f"⚠️ No attention weights for sample {i}")
                continue

            ax = axs[count]
            sns.heatmap(
                attention_weights[:len(pred_decoded), :len(src_decoded)].squeeze(1),
                xticklabels=src_decoded,
                yticklabels=pred_decoded,
                cmap="viridis",
                cbar=True,
                ax=ax
            )
            ax.set_xlabel("Source Tokens")
            ax.set_ylabel("Predicted Tokens")
            ax.set_title(f"Sample {i+1}")
            count += 1
            if count == 9:
                break

    plt.tight_layout()

    # Log to W&B if needed
    if wandb_log:
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        wandb.log({"Attention Heatmaps": wandb.Image(buf, caption="9 Sample Attention Heatmaps")})
        buf.close()

    plt.show()

def beam_search_decode_with_attention(model, src_tensor, src_vocab, tgt_vocab, beam_width, max_len, device):
    # Encode
    encoder_outputs, hidden = model.encoder(src_tensor)
    
    beams = [([tgt_vocab.sos_idx], 0.0, hidden, [])]  # seq, score, hidden, attention_list
    completed = []

    for _ in range(max_len):
        new_beams = []
        for seq, score, hidden_state, attn_seq in beams:
            if seq[-1] == tgt_vocab.eos_idx:
                completed.append((seq, score, attn_seq))
                continue

            input_token = torch.tensor([seq[-1]], device=device)
            output, hidden_next, attention = model.decoder(input_token, hidden_state, encoder_outputs, return_attention=True)

            log_probs = torch.log_softmax(output.squeeze(1), dim=-1)
            topk_log_probs, topk_indices = log_probs.topk(beam_width)

            for i in range(beam_width):
                token = topk_indices[0,i].item()
                log_prob = topk_log_probs[0,i].item()
                new_seq = seq + [token]
                new_attn_seq = attn_seq + [attention.squeeze(0).cpu()]
                new_beams.append((new_seq, score + log_prob, hidden_next, new_attn_seq))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        if all(seq[-1] == tgt_vocab.eos_idx for seq, _, _, _ in beams):
            completed.extend(beams)
            break

    # Best sequence
    best_seq, _, attn_seq = sorted(completed, key=lambda x: x[1], reverse=True)[0]
    attn_tensor = torch.stack(attn_seq)  # (tgt_len, src_len)

    return best_seq, attn_tensor