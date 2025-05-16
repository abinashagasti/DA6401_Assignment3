import torch
from torch import nn, optim
import os, wandb, yaml

from data_preprocess import *
from model import *
from utils import *

def train():
    wandb.init() # Initialize wandb run
    # wandb.init(resume="allow")
    config = wandb.config # Config for wandb sweep

    # Experiment name
    wandb.run.name = f"lr_{config.learning_rate}_#emb_{config.encoder_embedding_size}_{config.decoder_embedding_size}_#layers_{config.num_layers}_\
    cell_{config.cell_type}_hs_{config.hidden_layer_size}_bs_{config.batch_size}_drop_{config.dropout_prob}_teacher_{config.teacher_forcing}"

    # Configs
    data_dir = 'dakshina_dataset_v1.0'
    lang = 'hi'  # Hindi
    subfolder_dir = 'lexicons'
    train_path = os.path.join(data_dir, lang, subfolder_dir, f'{lang}.translit.sampled.train.tsv')
    dev_path = os.path.join(data_dir, lang, subfolder_dir, f'{lang}.translit.sampled.dev.tsv')
    test_path = os.path.join(data_dir, lang, subfolder_dir, f'{lang}.translit.sampled.test.tsv')

    encoder_embedding_dim = config.encoder_embedding_size
    decoder_embedding_dim = config.decoder_embedding_size
    hidden_dim = config.hidden_layer_size
    num_encoder_layers = config.num_layers
    num_decoder_layers = config.num_layers
    rnn_type = config.cell_type # can be 'RNN' or 'GRU'
    batch_size = config.batch_size
    num_epochs = 20
    learning_rate = config.learning_rate
    dropout_prob = config.dropout_prob
    teacher_forcing_ratio = config.teacher_forcing

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Get dataloaders
    train_loader, val_loader, _, src_vocab, tgt_vocab = prepare_dataloaders(train_path, dev_path, test_path, batch_size, repeat_datapoints=True, num_workers=4)

    # Initialize models
    encoder = Encoder(input_dim=len(src_vocab), emb_dim=encoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_encoder_layers, rnn_type=rnn_type, dropout=dropout_prob).to(device)
    decoder = Decoder(output_dim=len(tgt_vocab), emb_dim=decoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_decoder_layers, rnn_type=rnn_type, dropout=dropout_prob).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    train_model(model, train_loader, val_loader, optimizer, criterion, src_vocab, tgt_vocab, device, scheduler, num_epochs, teacher_forcing_ratio=teacher_forcing_ratio, accuracy_mode='both', wandb_log=True)

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    with open("sweep1.yaml", "r") as file:
        sweep_config = yaml.safe_load(file) # Read yaml file to store hyperparameters 

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment_3")
    entity = "ee20d201-indian-institute-of-technology-madras"
    project = "DA6401_Assignment_3"
    # sweep_id = "5t7ap5rq" # sweep1.yaml
    # sweep_id = "86x4jb7r" # sweep2.yaml
    # sweep_id = "j1h5tb43" # sweep3.yaml
    # sweep_id = "k9ldb4jj" # sweep1_100.yaml
    # sweep_id = "ex1e7bbi" # sweep4.yaml, finetuning sweep
    # api = wandb.Api() 

    # sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # if len(sweep.runs) >= 10:
    #     api.stop_sweep(sweep.id)
    #     print(f"Sweep {sweep.id} stopped after {len(sweep.runs)} runs.")

    # Start sweep agent
    wandb.agent(sweep_id, function=train, count=15, project=project)  # Run 10 experiments