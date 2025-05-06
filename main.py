import torch
from torch import nn, optim
import os

from data_preprocess import *
from model import *
from utils import *

def main():
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
    num_layers = 1
    rnn_type = 'LSTM'  # can be 'RNN' or 'GRU'
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.0001

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
    print(device)

    # Get dataloaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = prepare_dataloaders(train_path, dev_path, test_path, batch_size, repeat_datapoints=True)

    # Initialize models
    encoder = Encoder(input_dim=len(src_vocab), emb_dim=encoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_layers, rnn_type=rnn_type).to(device)
    decoder = Decoder(output_dim=len(tgt_vocab), emb_dim=decoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_layers, rnn_type=rnn_type).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_model(model, train_loader, val_loader, optimizer, criterion, tgt_vocab, device, num_epochs)

if __name__ == '__main__':
    main()