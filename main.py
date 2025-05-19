import torch
from torch import nn, optim
import os

from data_preprocess import *
from model import *
from utils import *

def main(mode: str = 'train', wandb_log: bool = False):
    user = "ee20d201-indian-institute-of-technology-madras"
    project = "DA6401_Assignment_3"
    display_name = "visualisation_table_attention"

    if wandb_log:
        wandb.init(entity=user, project=project, name=display_name)
        # wandb.run.name = display_name
    
    # Configs
    data_dir = 'dakshina_dataset_v1.0'
    lang = 'hi'  # Hindi
    subfolder_dir = 'lexicons'
    train_path = os.path.join(data_dir, lang, subfolder_dir, f'{lang}.translit.sampled.train.tsv')
    dev_path = os.path.join(data_dir, lang, subfolder_dir, f'{lang}.translit.sampled.dev.tsv')
    test_path = os.path.join(data_dir, lang, subfolder_dir, f'{lang}.translit.sampled.test.tsv')

    encoder_embedding_dim = 32
    decoder_embedding_dim = 128
    hidden_dim = 128
    num_encoder_layers = 2
    num_decoder_layers = 2
    rnn_type = 'LSTM'  # can be 'RNN' or 'LSTM' or 'GRU'
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.005
    dropout_prob = 0.2
    use_attention = True
    if use_attention:
        output_dir = 'predictions_attention'
    else:
        output_dir = 'predictions_vanilla'
    teacher_forcing_ratio = 0.75

    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Get dataloaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = prepare_dataloaders(train_path, dev_path, test_path, batch_size, repeat_datapoints=True, num_workers=4)

    # Initialize models
    encoder = Encoder(input_dim=len(src_vocab), emb_dim=encoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_encoder_layers, rnn_type=rnn_type, dropout=dropout_prob).to(device)
    decoder = Decoder(output_dim=len(tgt_vocab), emb_dim=decoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_decoder_layers, rnn_type=rnn_type, dropout=dropout_prob, use_attention=True).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    if mode == 'train':
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # Training loop
        train_model(model, train_loader, val_loader, optimizer, criterion, src_vocab, tgt_vocab, device, scheduler,
                     num_epochs, teacher_forcing_ratio=teacher_forcing_ratio, accuracy_mode='both', patience=7, wandb_log=wandb_log, beam_validate=True, beam_width=3)
    
    elif mode == 'test':
        # Load checkpoint
        checkpoint = torch.load('best_att.pth', map_location=device, weights_only=True)
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        # test_model_alternate(model, test_loader, criterion, src_vocab, tgt_vocab, device, beam_validate=True, output_dir=output_dir, wandb_log=wandb_log, n=20)
        plot_attention_heatmaps(model, test_loader, tgt_vocab, src_vocab, device)

    else:
        raise ValueError(f"mode = {mode} \nMode should be a string taking value either 'train' or 'test'.")
    
    if wandb_log:
        wandb.finish()

if __name__ == '__main__':
    mode = 'test'
    wandb_log = False
    main(mode, wandb_log) 