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


def predict_with_attention(model, src, sos_idx, eos_idx, max_len=30):
    '''
    src: (1, src_len) – tokenized input sequence
    Returns:
        - predictions: list of token IDs (excluding <sos>)
        - attention_weights: list of attention tensors, one per target step (each: [1, src_len])
    '''
    model.eval()
    predictions = []
    attentions = []

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        hidden = model.match_encoder_decoder_hidden(hidden, model.decoder.num_layers)

        input_token = torch.tensor([sos_idx], device=model.device)  # start with <sos>

        for _ in range(max_len):
            output, hidden, attn = model.decoder(input_token, hidden, encoder_outputs, return_attention=True)

            top1 = output.argmax(1).item()
            if top1 == eos_idx:
                break
            predictions.append(top1)
            attentions.append(attn.squeeze(1).cpu())  # remove (1) dimension, shape: [src_len]

            input_token = torch.tensor([top1], device=model.device)

    # Stack attentions into a (tgt_len, src_len) tensor
    attn_tensor = torch.stack(attentions) if attentions else torch.empty(0)
    return predictions, attn_tensor  # both on CPU

def plot_attention_heatmaps(model, test_loader, tgt_vocab, src_vocab, device='cpu'):
    """
    Plots attention heatmaps for 10 sampled test sequences.
    
    model: the trained Seq2Seq model
    test_loader: DataLoader with test data
    tgt_vocab: target vocab object (with idx_to_token and eos_idx, sos_idx)
    src_vocab: optional, if you want to decode source indices to tokens
    device: 'cpu' or 'cuda'
    """
    model.eval()
    sampled_src, _ = sample_from_testset(test_loader)

    for i, src_tensor in enumerate(sampled_src):
        src = src_tensor.unsqueeze(0).to(device)
        predicted_ids, attn_tensor = predict_with_attention(model, src, tgt_vocab.sos_idx, tgt_vocab.eos_idx)

        # Decode tokens (optional)
        if src_vocab:
            src_tokens = [src_vocab.idx_to_token[idx] for idx in src_tensor.tolist()]
        else:
            src_tokens = [str(idx.item()) for idx in src_tensor]  # fallback: just index

        tgt_tokens = [tgt_vocab.idx_to_token[idx] for idx in predicted_ids]

        print(attn_tensor.squeeze(1).cpu().numpy())

        # attn_tensor: (tgt_len, src_len)
        plt.figure(figsize=(10, 6))
        sns.heatmap(attn_tensor.squeeze(1).cpu().numpy(), 
                    xticklabels=src_tokens, 
                    yticklabels=tgt_tokens, 
                    cmap="viridis", 
                    cbar=True, 
                    linewidths=0.5, 
                    annot=True, 
                    fmt=".2f")

        plt.xlabel("Source tokens")
        plt.ylabel("Predicted tokens")
        plt.title(f"Attention Heatmap {i+1}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()