import torch
import torch.nn as nn

class Encoder(nn.Module):
    '''
    This class implements the encoder module for a sequence-to-sequence model.

    Inputs:
        - input_dim: vocabulary size of the source language
        - emb_dim: dimension of the character embeddings
        - hidden_dim: dimension of the hidden state in the RNN
        - num_layers: number of RNN layers (default = 1)
        - rnn_type: one of 'rnn', 'lstm', 'gru' (default = 'lstm')
        - dropout: dropout probability between layers (default = 0.0)

    Outputs:
        - outputs: hidden states for each time step (batch_size, seq_len, hidden_dim)
        - hidden: final hidden state(s), shape depends on RNN type
    '''
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1, rnn_type='lstm', dropout=0.0, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p = dropout)
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

    def forward(self, input):
        # src shape: (batch_size, src_seq_len)
        embedded = self.dropout(self.embedding(input))  # shape: (batch_size, src_seq_len, emb_dim)
        outputs, hidden = self.rnn(embedded)  # outputs: (batch_size, src_seq_len, hidden_dim)
        return outputs, hidden

class Decoder(nn.Module):
    '''
    This class implements the decoder module for a sequence-to-sequence model.

    Inputs:
        - output_dim: vocabulary size of the target language
        - emb_dim: dimension of the character embeddings
        - hidden_dim: dimension of the hidden state in the RNN
        - num_layers: number of RNN layers (default = 1)
        - rnn_type: one of 'rnn', 'lstm', 'gru' (default = 'lstm')
        - dropout: dropout probability between layers (default = 0.0)

    Outputs:
        - output: predicted logits over vocabulary (batch_size, output_dim)
        - hidden: final hidden state(s), shape depends on RNN type
    '''
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1, rnn_type='lstm', dropout=0.0, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p = dropout)
        self.rnn_type = rnn_type.lower()
        self.output_dim = output_dim

        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden):
        '''
        input: (batch_size) –  current input token ids
        hidden: hidden state(s) from previous time step

        Returns:
            output: (batch_size, output_dim) – logits over vocabulary
            hidden: updated hidden state(s)
        '''
        input = input.unsqueeze(1)  # (batch_size) → (batch_size, 1)
        embedded = self.dropout(self.embedding(input))  # (batch_size, 1, emb_dim)

        output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, 1, hidden_dim)
        prediction = self.fc_out(output.squeeze(1))  # (batch_size, output_dim)

        return prediction, hidden


class Seq2Seq(nn.Module):
    '''
    Combines the Encoder and Decoder into one Seq2Seq model.

    Inputs:
        encoder: Encoder model
        decoder: Decoder model
        device: torch.device ('cpu' or 'cuda')

    Methods:
        forward(src, tgt, teacher_forcing_ratio): training forward pass
        inference(src, max_len): inference for a single source sequence
    '''
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=1.0):
        '''
        Forward pass for training.
        src: (batch_size, src_len)
        tgt: (batch_size, tgt_len)
        '''
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(batch_size, tgt_len, output_dim).to(self.device)

        # Encoder forward
        encoder_outputs, hidden = self.encoder(src)

        if self.encoder.rnn_type == 'lstm':
            hidden_state, cell = hidden
            hidden = (hidden_state[-1:].contiguous(), cell[-1:].contiguous())
        else:
            hidden = hidden[-1:].contiguous()

        # First input to decoder is <sos>
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = tgt[:, t] if teacher_force else top1

        return outputs

    def inference(self, src, sos_idx, eos_idx, max_len=30):
        '''
        Inference (no teacher forcing).

        src: (1, src_len) — single sequence
        Returns list of predicted token indices
        '''
        self.eval()

        with torch.no_grad():
            _, hidden = self.encoder(src)
            input_token = torch.tensor([sos_idx], device=self.device)

            outputs = []

            for _ in range(max_len):
                output, hidden = self.decoder(input_token, hidden)
                top1 = output.argmax(1).item()
                if top1 == eos_idx:
                    break
                outputs.append(top1)
                input_token = torch.tensor([top1], device=self.device)

        return outputs