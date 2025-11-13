import pandas as pd
import numpy as np
import re
import string
import nltk
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def load_data(file_path='../data/IMDB Dataset.csv') -> pd.DataFrame:
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Preprocess - Clean, tokenize, prepare dataset dict
    """

	# Convert sentiment to binary
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    train_df = df.iloc[:25000].reset_index(drop=True)
    test_df = df.iloc[25000:].reset_index(drop=True)

	# Clean and tokenize text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())

    train_df['tokens'] = train_df['review'].apply(clean_text).apply(nltk.word_tokenize)
    test_df['tokens'] = test_df['review'].apply(clean_text).apply(nltk.word_tokenize)

	# Build vocabulary with only top 10k tokens
    all_tokens = [token for tokens in train_df['tokens'] for token in tokens]
    word_counts = Counter(all_tokens)
    most_common = word_counts.most_common(10000 - 2) # -2 for <PAD> and <UNK>

    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        word2idx[word] = idx

	# Prepare datasets with different sequence lengths
    def prepare_dataset(df: pd.DataFrame, seq_length):
        sequences = []
        for tokens in df['tokens']:
            seq = [word2idx.get(token, 1) for token in tokens]
            if len(seq) > seq_length:
                seq = seq[:seq_length]
            else:
                seq = seq + [0] * (seq_length - len(seq))
            sequences.append(seq)

        return np.array(sequences, dtype=np.int32), df['label'].values.astype(np.float32)

    datasets = {}
    for seq_len in [25, 50, 100]:
        X_train, y_train = prepare_dataset(train_df, seq_len)
        X_test, y_test = prepare_dataset(test_df, seq_len)
        datasets[seq_len] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }

    return datasets, word2idx
