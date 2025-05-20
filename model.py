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

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx) # embedding layer
        self.dropout = nn.Dropout(p = dropout) # dropout layer after embedding
        self.rnn_type = rnn_type.lower() # rnn type 
        self.num_layers = num_layers # number of hidden layers in the encoder

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
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1, rnn_type='lstm', dropout=0.0, pad_idx=0, use_attention=False):
        super().__init__()

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx) # embedding layer of decoder
        self.dropout = nn.Dropout(p = dropout) # dropout layer
        self.rnn_type = rnn_type.lower() # rnn type
        self.output_dim = output_dim # output vocabulary size of target
        self.num_layers = num_layers # number of hidden layers in the decoder
        self.use_attention = use_attention # bool denoting whether the decoder uses attention

        self.attention = Attention(hidden_dim)  # Attention module
        # If attention is used, we'll concatenate context vector with embedded input
        input_dim = emb_dim + hidden_dim if self.use_attention else emb_dim

        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs=None, return_attention=False):
        '''
        input: (batch_size) –  current input token ids
        hidden: hidden state(s) from previous time step

        Returns:
            output: (batch_size, output_dim) – logits over vocabulary
            hidden: updated hidden state(s)
        '''
        input = input.unsqueeze(1)  # (batch_size) → (batch_size, 1)
        embedded = self.dropout(self.embedding(input))  # (batch_size, 1, emb_dim)

        if not self.use_attention:
            output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, 1, hidden_dim)
            prediction = self.fc_out(output.squeeze(1))  # (batch_size, output_dim)
        else:
            if isinstance(hidden, tuple):  # LSTM
                last_hidden = hidden[0][-1]  # (batch_size, hidden_dim)
            else:  # GRU/RNN
                last_hidden = hidden[-1]

            attn_weights = self.attention(last_hidden, encoder_outputs)  # (batch_size, src_len)
            attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, src_len)

            context = torch.bmm(attn_weights, encoder_outputs)  # (batch_size, 1, hidden_dim)

            rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, emb_dim + hidden_dim)

            output, hidden = self.rnn(rnn_input, hidden)
            prediction = self.fc_out(output.squeeze(1))  # (batch_size, output_dim)

        if return_attention:
            return prediction, hidden, attn_weights
        else:
            return prediction, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        src_len = encoder_outputs.size(1)

        # Repeat hidden across src_len
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch_size, src_len, hidden_dim)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # (batch_size, src_len, hidden_dim)
        attention = self.v(energy).squeeze(2)  # (batch_size, src_len)
        return torch.softmax(attention, dim=1)  # (batch_size, src_len)

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

        # Make encoder hidden output compatible with decoder if num_layers are different in both networks
        hidden = self.match_encoder_decoder_hidden(hidden, self.decoder.num_layers)

        # First input to decoder is <sos>
        input_token = tgt[:, 0]

        for t in range(1, tgt_len):
            if self.decoder.use_attention:
                output, hidden = self.decoder(input_token, hidden, encoder_outputs) # send encoder outputs if decoder uses attention
            else:
                output, hidden = self.decoder(input_token, hidden) # encoder outputs not required if decoder does not use attention
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio # bool denoting if teacher forcing is active
            top1 = output.argmax(1) # best predicted output token

            input_token = tgt[:, t] if teacher_force else top1 
            # feed target token if teacher_forcing is enabled else feed previous predicted output

        return outputs
    
    def adjust_layers(self, h, n_decoder):
        # helps adjust the hidden output if number of encoder and decoder hidden layers are different
        n_encoder = h.size(0)
        if n_encoder == n_decoder:
            return h
        elif n_encoder < n_decoder:
            # if lesser number of encoder layers are there then repeat last hidden layer of encoder
            repeat_h = h[-1:].repeat(n_decoder - n_encoder, 1, 1)
            return torch.cat([h, repeat_h], dim=0)
        else:
            # if lesser number of decoder layers than encoder then send last few hidden layers of encoder
            return h[-n_decoder:]
    
    def match_encoder_decoder_hidden(self, hidden, num_decoder_layers):
        # separately treat lstm and rnn/gru layers
        if isinstance(hidden, tuple):
            # LSTM: hidden is (h_n, c_n)
            h_n, c_n = hidden
            h_n = self.adjust_layers(h_n, num_decoder_layers)
            c_n = self.adjust_layers(c_n, num_decoder_layers)
            return (h_n, c_n)
        else:
            # GRU/RNN: hidden is just h_n
            return self.adjust_layers(hidden, num_decoder_layers)

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