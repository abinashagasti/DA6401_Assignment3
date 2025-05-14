import torch
from torch.utils.data import Dataset, DataLoader
import csv
from functools import partial

# Tokens apart from the alphabet
PAD_TOKEN = "<PAD>" # denotes a padding element in the character sequence
SOS_TOKEN = "<GO>" # denotes start of sequence token for decoder model
EOS_TOKEN = "<STOP>" # denotes end of sequence token for decoder model
UNK_TOKEN = "<UNK>" # token for unknown characters

class Vocab:
    # Constructor
    def __init__(self, tokens=None):
        self.token_to_idx = {} # dictionary containing character tokens as keys and an assigned index as value
        self.idx_to_token = [] # list with character token in its assigned index
        self.special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] # set of special tokens apart from the alphabet

        # add special tokens into vocabulary
        for token in self.special_tokens:
            self.add_token(token)

        self.pad_idx = self.token_to_idx[PAD_TOKEN]

        # add tokens passed into the constructor function
        if tokens:
            for token in tokens:
                self.add_token(token)

    # Methods
                
    # Add token to vocab dictionary and list if encountered for the first time
    def add_token(self, token):
        if token not in self.token_to_idx:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def add_string(self, string):
        for character in string:
            self.add_token(character)

    # Returns total number of tokens in vocabulary
    def __len__(self):
        return len(self.idx_to_token)

    # Returns index of given token
    def token2idx(self, token):
        return self.token_to_idx.get(token, self.token_to_idx[UNK_TOKEN])

    # Returns token given an index
    def idx2token(self, idx):
        return self.idx_to_token[idx]

    # Encodes given character sequence into list of indices
    def encode(self, sequence, add_sos_eos=False):
        tokens = list(sequence) # converts character sequence into list of character tokens
        indices = [self.token2idx(tok) for tok in tokens] # obtains list of indices corresponding to the character tokens
        if add_sos_eos:
            # adds sos and eos token indices in the beginning and end if specified
            indices = [self.token2idx(SOS_TOKEN)] + indices + [self.token2idx(EOS_TOKEN)]
        return indices

    # Decodes given list of indices into character tokens
    def decode(self, indices, remove_special=True):
        tokens = [self.idx2token(idx) for idx in indices] # list of tokens corresponding to each index
        if remove_special:
            # removes special tokens if specified
            tokens = [t for t in tokens if t not in self.special_tokens]
        return "".join(tokens) # returns character sequence combining the tokens
    
 # Dataset class to be passed to a DataLoader   
class TransliterationDataset(Dataset):
    # Constructor function
    def __init__(self, path, src_vocab, tgt_vocab, add_sos_eos=True, repeat_datapoints=True):
        # set source and target Vocab objects
        self.src_vocab = src_vocab 
        self.tgt_vocab = tgt_vocab
        self.add_sos_eos = add_sos_eos # flag controlling addition of sos and eos tokens in target encodings
        self.data = [] # list containing all data pairs

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t') # read from tsv file and extract data
            for row in reader:
                if len(row) != 3:
                    continue
                tgt, src, count = row # extract target devanagari script and source latin script
                if not repeat_datapoints:
                    count = 1
                src_ids = src_vocab.encode(src, add_sos_eos=False) # encode source sequence
                tgt_ids = tgt_vocab.encode(tgt, add_sos_eos=add_sos_eos) # encode target sequence
                for _ in range(int(count)):
                    self.data.append((src_ids, tgt_ids)) # append source and target encodings into self.data the number of times

    # Methods returning length and fetching item
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def collate_fn(batch, source_pad_idx, target_pad_idx):
    '''
    This function takes a batch of (source, target) pairs,
    pads the source and target sequences separately to the length of the longest sequence in the batch,
    and returns a dictionary containing the padded source and target tensors.

    Inputs:
        batch: List of samples, where each sample is a dictionary with keys 'source' and 'target',
            and each value is a 1D tensor of token indices.
        source_pad_idx: token assigned to padding in source vocab
        target_pad_idx: token assigned to padding in target vocab

    Outputs:
        A dictionary with two keys:
            - 'source': a tensor of shape (batch_size, max_source_length) with padded source sequences
            - 'target': a tensor of shape (batch_size, max_target_length) with padded target sequences
    '''
    src_batch, tgt_batch = zip(*batch) # get source and target batches separately from input dataset batch

    # src_lens = [len(x) for x in src_batch] # length of sequences in source batch
    # tgt_lens = [len(x) for x in tgt_batch] # length of sequences in target batch

    src_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq) for seq in src_batch],
        batch_first=True,
        padding_value=source_pad_idx
    ) # pad source batch with padding_value=source_pad_idx and first dimension corresponding to batch_size
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq) for seq in tgt_batch],
        batch_first=True,
        padding_value=target_pad_idx
    ) # pad target batch with padding_value=target_pad_idx and first dimension corresponding to batch_size

    return src_padded, tgt_padded #, src_lens, tgt_lens

def prepare_dataloaders(train_file_path, val_file_path, test_file_path, batch_size=32, repeat_datapoints=True, num_workers=4):
    '''
    This function prepares the train, validation, and test DataLoaders for the transliteration task.

    Steps:
        1. Builds the source and target vocabularies from the training data.
        2. Creates TransliterationDataset instances for train, val, and test splits.
        3. Creates corresponding DataLoaders using a custom collate function to pad sequences.

    Inputs:
        - train_file_path (str): path to the training data (tsv file).
        - val_file_path (str): path to the validation data (tsv file).
        - test_file_path (str): path to the test data (tsv file).
        - batch_size (int): batch size for the DataLoaders (default: 32).
        - repeat_datapoints (bool): retain duplicates in dataset (default: True).
        - num_workers (int): number of cpu cores allotted (default: 4). 

    Outputs:
        - train_loader (DataLoader): DataLoader for training data
        - val_loader (DataLoader): DataLoader for validation data
        - test_loader (DataLoader): DataLoader for test data
        - source_vocab (Vocab): Vocabulary object for the source language
        - target_vocab (Vocab): Vocabulary object for the target language
    '''

    # initialise source and target vocabularies
    source_vocab = Vocab()
    target_vocab = Vocab()
    
    # Build vocab from training data
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            target, source, _ = line.strip().split('\t')
            source_vocab.add_string(source)
            target_vocab.add_string(target)

    # Create datasets
    train_dataset = TransliterationDataset(train_file_path, source_vocab, target_vocab, repeat_datapoints=repeat_datapoints)
    val_dataset = TransliterationDataset(val_file_path, source_vocab, target_vocab)
    test_dataset = TransliterationDataset(test_file_path, source_vocab, target_vocab)

    collate = partial(collate_fn, source_pad_idx=source_vocab.pad_idx, target_pad_idx = target_vocab.pad_idx)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate
        # collate_fn=lambda batch: collate_fn(batch, source_vocab.pad_idx, target_vocab.pad_idx)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate
        # collate_fn=lambda batch: collate_fn(batch, source_vocab.pad_idx, target_vocab.pad_idx)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate
        # collate_fn=lambda batch: collate_fn(batch, source_vocab.pad_idx, target_vocab.pad_idx)
    )

    return train_loader, val_loader, test_loader, source_vocab, target_vocab