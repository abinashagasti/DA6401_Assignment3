import torch
from torch import nn, optim
import os

from data_preprocess import *
from model import *
from utils import *

def main(mode: str = 'train', wandb_log: bool = False):
    user = "ee20d201-indian-institute-of-technology-madras"
    project = "DA6401_Assignment_3"
    display_name = "test_run"

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

    encoder_embedding_dim = 30
    decoder_embedding_dim = 67
    hidden_dim = 128
    num_encoder_layers = 1
    num_decoder_layers = 1
    rnn_type = 'LSTM'  # can be 'RNN' or 'GRU'
    batch_size = 64
    num_epochs = 2
    learning_rate = 0.01
    dropout_prob = 0.3

    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Get dataloaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = prepare_dataloaders(train_path, dev_path, test_path, batch_size, repeat_datapoints=True, num_workers=4)

    # Initialize models
    encoder = Encoder(input_dim=len(src_vocab), emb_dim=encoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_encoder_layers, rnn_type=rnn_type, dropout=dropout_prob).to(device)
    decoder = Decoder(output_dim=len(tgt_vocab), emb_dim=decoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_decoder_layers, rnn_type=rnn_type, dropout=dropout_prob).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    if mode == 'train':
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # Training loop
        train_model(model, train_loader, val_loader, optimizer, criterion, src_vocab, tgt_vocab, device, scheduler,
                     num_epochs, teacher_forcing_ratio=None, accuracy_mode='both', patience=7, wandb_log=wandb_log, beam_validate=False)
    
    elif mode == 'test':
        # Load checkpoint
        checkpoint = torch.load('best_model.pth', map_location=device, weights_only=True)

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        test_model(model, test_loader, src_vocab, tgt_vocab, device, beam_width=3, output_dir="predictions_vanilla")

    else:
        raise ValueError(f"mode = {mode} \nMode should be a string taking value either 'train' or 'test'.")
    
    if wandb_log:
        wandb.finish()

if __name__ == '__main__':
    mode = 'train'
    wandb_log = False
    main(mode, wandb_log) 