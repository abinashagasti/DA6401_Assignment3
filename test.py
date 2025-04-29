from data_preprocess import *

train_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv'
val_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv'
test_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv'
train_loader, val_loader, test_loader, source_vocab, target_vocab = prepare_dataloaders(train_path, val_path, test_path, batch_size=64)