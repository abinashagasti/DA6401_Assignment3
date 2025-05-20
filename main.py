import torch
from torch import nn, optim
import os
import argparse

from data_preprocess import *
from model import *
from utils import *

def main(args):
    user = "ee20d201-indian-institute-of-technology-madras"
    project = "DA6401_Assignment_3"
    display_name = args.wandb_display_name

    if args.wandb_log:
        wandb.init(entity=user, project=project, name=display_name)
        # wandb.run.name = display_name
    
    # Configure hyperparameters
    data_dir = args.data_directory#'dakshina_dataset_v1.0'
    lang = args.language #'hi'  # Hindi
    subfolder_dir = 'lexicons'
    train_path = os.path.join(data_dir, lang, subfolder_dir, f'{lang}.translit.sampled.train.tsv')
    dev_path = os.path.join(data_dir, lang, subfolder_dir, f'{lang}.translit.sampled.dev.tsv')
    test_path = os.path.join(data_dir, lang, subfolder_dir, f'{lang}.translit.sampled.test.tsv')

    encoder_embedding_dim = args.encoder_embedding_dim # 32
    decoder_embedding_dim = args.decoder_embedding_dim # 128
    hidden_dim = args.hidden_dim # 128
    num_encoder_layers = args.num_encoder_layers # 2
    num_decoder_layers = args.num_decoder_layers # 2
    rnn_type = args.cell_type  # can be 'RNN' or 'LSTM' or 'GRU'
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    dropout_prob = args.dropout
    use_attention = args.use_attention
    if use_attention:
        output_dir = 'predictions_attention'
    else:
        output_dir = 'predictions_vanilla'
    teacher_forcing_ratio = args.teacher_forcing

    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Get dataloaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = prepare_dataloaders(train_path, dev_path, test_path, batch_size, repeat_datapoints=True, num_workers=1)

    # Initialize models
    encoder = Encoder(input_dim=len(src_vocab), emb_dim=encoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_encoder_layers, rnn_type=rnn_type, dropout=dropout_prob).to(device)
    decoder = Decoder(output_dim=len(tgt_vocab), emb_dim=decoder_embedding_dim, hidden_dim=hidden_dim,
                      num_layers=num_decoder_layers, rnn_type=rnn_type, dropout=dropout_prob, use_attention=True).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    if args.mode == 'train':
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # Training loop
        train_model(model, train_loader, val_loader, optimizer, criterion, src_vocab, tgt_vocab, device, scheduler,
                     num_epochs, teacher_forcing_ratio=teacher_forcing_ratio, accuracy_mode='both', patience=7, wandb_log=args.wandb_log, beam_validate=True, beam_width=3)
    
    elif args.mode == 'test':
        # Load checkpoint
        if use_attention:
            checkpoint = torch.load('best_att.pth', map_location=device, weights_only=True)
        else:
            checkpoint = torch.load('best.pth', map_location=device, weights_only=True)
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        test_model_alternate(model, test_loader, criterion, src_vocab, tgt_vocab, device, beam_validate=True, output_dir=output_dir, wandb_log=args.wandb_log, n=20)
        # plot_attention_heatmaps(model, test_loader, tgt_vocab, src_vocab, device, 10, args.wandb_log)
        # plot_connectivity(model, src_vocab, tgt_vocab, filepath='predictions_attention/predictions.txt', device=device, wandb_log=args.wandb_log)

    else:
        raise ValueError(f"mode = {args.mode} \nMode should be a string taking value either 'train' or 'test'.")
    
    if args.wandb_log:
        wandb.finish()

# if __name__ == '__main__':
#     mode = 'test'
#     wandb_log = True
#     main(mode, wandb_log) 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp","--wandb_project",default="DA6401_Assignment_2", help="Project name used to track experiments in Weights & Biases dashboard",type=str)
    parser.add_argument("-we","--wandb_entity",default="ee20d201-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard",type=str)
    parser.add_argument("-d","--data_directory",default="dakshina_dataset_v1.0",type=str,help="Path containing dakshina dataset.")
    parser.add_argument("-name","--wandb_display_name",default="test_run",type=str,help="Wandb run display name.")
    parser.add_argument("-l","--language",default="hi",type=str,help="Language of transliteration.")
    parser.add_argument("-e","--epochs",default=20,help="Number of epochs to train neural network",type=int)
    parser.add_argument("-enc","--encoder_embedding_dim",default=32,help="Encoder embedding dimension",type=int)
    parser.add_argument("-dec","--decoder_embedding_dim",default=128,help="Decoder embedding dimension",type=int)
    parser.add_argument("-hid","--hidden_dim",default=128,help="Hidden dimension",type=int)
    parser.add_argument("-b","--batch_size",default=32,help="Batch size used to train neural network",type=int)
    parser.add_argument("-da","--use_attention",default=True,action="store_false",help="Use data augmentation for training.")
    parser.add_argument("-nel","--num_encoder_layers",default=2,help="Number of hidden encoder layers.",type=int)
    parser.add_argument("-ndl","--num_decoder_layers",default=2,help="Number of hidden decoder layers.",type=int)
    parser.add_argument("-drprob","--dropout",default=0.2,type=float,help="Dropout prob.")
    parser.add_argument("-teach","--teacher_forcing",default=0.75,help="Padding used in each layer.",type=float)
    parser.add_argument("-lr","--learning_rate",default=0.005, help="Learning rate used to optimize model parameters",type=float)
    parser.add_argument("-w_d","--weight_decay",default=0.0001, help="Weight decay used by optimizers",type=float)
    parser.add_argument("-m","--mode",default='train',choices = ['train','test'], help="Activation functions",type=str)
    parser.add_argument("-cell","--cell_type",default='lstm',choices = ['rnn','lstm','gru'], help="Activation functions",type=str)
    parser.add_argument("-wbl","--wandb_log",default=False,action="store_true", help="Login data onto wandb.ai")
    args = parser.parse_args()
    main(args)