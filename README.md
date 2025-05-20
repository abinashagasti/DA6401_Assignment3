# DA6401 - Assignment 3

This assignment implements a sequence-to-sequence (seq2seq) model in PyTorch. It includes data preprocessing, vocabulary creation, training/testing routines, and visualizations for understanding the task of transliteration.

---

## File Descriptions

### `main.py`
The **main entry point** of the project. It:
- Loads and preprocesses the data
- Initializes the model, loss, and optimizer
- Runs the training and evaluation loops
- Supports visualizing attention weights
- Optionally logs metrics and visualizations to Weights & Biases (wandb)

### `data_preprocess.py`
- Builds and manages vocabularies, including special tokens like `<sos>`, `<eos>`, and `<pad>`
- Tokenizes text, prepares PyTorch datasets and dataloaders

### `model.py`
Defines the model architecture:

- `Encoder`: Embeds and encodes input sequences
- `Decoder`: Generates target sequences step-by-step
- `Attention`: Implements an attention mechanism to help the decoder focus on relevant parts of the input
- `Seq2Seq`: Wraps encoder and decoder into a complete model

### `utils.py`
Helper functions for:

- Training and evaluation loops
- Logging and checkpointing
- General utilities used throughout the codebase

### `visualisation.py`
- Aids in visualising multiple items for the assignment

---
## Command Line Arguments

You can configure the behavior of `main.py` using the following command-line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `-wp`, `--wandb_project` | Weights & Biases project name for logging experiments | `"DA6401_Assignment_3"` |
| `-we`, `--wandb_entity` | WandB entity/team under which the project runs | `"ee20d201-indian-institute-of-technology-madras"` |
| `-d`, `--data_directory` | Path to the `dakshina_dataset_v1.0` dataset | `"dakshina_dataset_v1.0"` |
| `-l`, `--language` | Language code for transliteration (e.g., `hi`, `bn`) | `"hi"` |
| `-e`, `--epochs` | Number of training epochs | `20` |
| `-enc`, `--encoder_embedding_dim` | Embedding dimension of the encoder | `32` |
| `-dec`, `--decoder_embedding_dim` | Embedding dimension of the decoder | `128` |
| `-hid`, `--hidden_dim` | Hidden layer dimension for both encoder and decoder | `128` |
| `-b`, `--batch_size` | Batch size used during training | `32` |
| `-da`, `--use_attention` | Disable attention (default is `True`, use `--no-use_attention` to turn it off) | `True` |
| `-nel`, `--num_encoder_layers` | Number of layers in the encoder | `2` |
| `-ndl`, `--num_decoder_layers` | Number of layers in the decoder | `2` |
| `-drprob`, `--dropout` | Dropout probability | `0.2` |
| `-teach`, `--teacher_forcing` | Probability of using teacher forcing during training | `0.75` |
| `-lr`, `--learning_rate` | Learning rate for the optimizer | `0.005` |
| `-w_d`, `--weight_decay` | Weight decay (L2 regularization) | `0.0001` |
| `-m`, `--mode` | Execution mode: `train` or `test` | `"train"` |
| `-cell`, `--cell_type` | Type of RNN cell to use: `rnn`, `lstm`, or `gru` | `"lstm"` |
| `-wbl`, `--wandb_log` | Enable Weights & Biases logging (`--wandb_log` to enable) | `False` |

### Example Usage

```bash
# Train using default settings
python3 main.py

# Train with GRU and WandB logging enabled
python3 main.py --cell_type gru --wandb_log

# Test using a custom model configuration
python3 main.py -m test --cell_type lstm --language hi

# Use `--help` to see all available options:
python3 main.py --help
```

