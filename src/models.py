import jax
import jax.numpy as jnp
import flax.linen as nn
import functools


ACTIVATION_FUNCTIONS = {
    'tanh': nn.tanh,
    'relu': nn.relu,
    'sigmoid': nn.sigmoid
}


def flip_sequences(inputs, lengths):
    """
    Flip sequences along time dimension for bidirectional processing
    """
    max_length = inputs.shape[1]
    indices = jnp.arange(max_length)
    reversed_indices = jnp.maximum(0, lengths[:, None] - 1 - indices)
    return jnp.take_along_axis(inputs, reversed_indices[:, :, None], axis=1)


class SimpleRNN(nn.Module):
    vocab_size: int
    embedding_dim: int
    hidden_size: int
    activation: str = 'tanh'
    dropout_rate: float = 0.4

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(x)

        activation_fn = ACTIVATION_FUNCTIONS[self.activation]
        rnn1 = nn.RNN(nn.SimpleCell(features=self.hidden_size, activation_fn=activation_fn))
        x = rnn1(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        rnn2 = nn.RNN(nn.SimpleCell(features=self.hidden_size, activation_fn=activation_fn))
        x = rnn2(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = x[:, -1, :]
        x = nn.Dense(features=1)(x)
        x = nn.sigmoid(x)
        return x.squeeze(-1)


class LSTMModel(nn.Module):
    vocab_size: int
    embedding_dim: int
    hidden_size: int
    activation: str = 'tanh'
    dropout_rate: float = 0.4

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(x)

        activation_fn = ACTIVATION_FUNCTIONS[self.activation]
        lstm1 = nn.RNN(nn.OptimizedLSTMCell(features=self.hidden_size, activation_fn=activation_fn))
        x = lstm1(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        lstm2 = nn.RNN(nn.OptimizedLSTMCell(features=self.hidden_size, activation_fn=activation_fn))
        x = lstm2(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)

        x = x[:, -1, :]
        x = nn.Dense(features=1)(x)
        x = nn.sigmoid(x)
        return x.squeeze(-1)


class ScannedLSTM(nn.Module):
    hidden_size: int
    activation_fn: callable = nn.tanh

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=1,
        out_axes=1,
        split_rngs={'params': False},
    )
    @nn.compact
    def __call__(self, carry, x):
        return nn.OptimizedLSTMCell(self.hidden_size, activation_fn=self.activation_fn)(carry, x)

    def initialize_carry(self, input_shape):
        return nn.OptimizedLSTMCell(self.hidden_size, activation_fn=self.activation_fn, parent=None).initialize_carry(
            jax.random.key(0), input_shape
        )


class BiLSTMModel(nn.Module):
    vocab_size: int
    embedding_dim: int
    hidden_size: int
    activation: str = 'tanh'
    dropout_rate: float = 0.4

    def setup(self):
        activation_fn = ACTIVATION_FUNCTIONS[self.activation]
        self.embed = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)
        self.forward_lstm1 = ScannedLSTM(self.hidden_size, activation_fn)
        self.backward_lstm1 = ScannedLSTM(self.hidden_size, activation_fn)
        self.forward_lstm2 = ScannedLSTM(self.hidden_size, activation_fn)
        self.backward_lstm2 = ScannedLSTM(self.hidden_size, activation_fn)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.dense = nn.Dense(features=1)

    def __call__(self, x, train=True):
        batch_size = x.shape[0]
        x = self.embed(x)
        seq_lengths = jnp.full((batch_size,), x.shape[1], dtype=jnp.int32)

        # BiLSTM Layer 1
        initial_state = self.forward_lstm1.initialize_carry(x[:, 0].shape)
        _, forward_out1 = self.forward_lstm1(initial_state, x)

        reversed_x = flip_sequences(x, seq_lengths)
        initial_state = self.backward_lstm1.initialize_carry(reversed_x[:, 0].shape)
        _, backward_out1 = self.backward_lstm1(initial_state, reversed_x)
        backward_out1 = flip_sequences(backward_out1, seq_lengths)

        x = jnp.concatenate([forward_out1, backward_out1], axis=-1)
        x = self.dropout(x, deterministic=not train)

        # BiLSTM Layer 2
        initial_state = self.forward_lstm2.initialize_carry(x[:, 0].shape)
        _, forward_out2 = self.forward_lstm2(initial_state, x)

        reversed_x2 = flip_sequences(x, seq_lengths)
        initial_state = self.backward_lstm2.initialize_carry(reversed_x2[:, 0].shape)
        _, backward_out2 = self.backward_lstm2(initial_state, reversed_x2)
        backward_out2 = flip_sequences(backward_out2, seq_lengths)

        x = jnp.concatenate([forward_out2, backward_out2], axis=-1)
        x = self.dropout(x, deterministic=not train)

        x = x[:, -1, :]
        x = self.dense(x)
        x = nn.sigmoid(x)
        return x.squeeze(-1)


def create_model(model_type, vocab_size, embedding_dim, hidden_size, activation, dropout_rate):
    """
    Create model based on type
    """
    if model_type == 'rnn':
        return SimpleRNN(vocab_size, embedding_dim, hidden_size, activation, dropout_rate)
    elif model_type == 'lstm':
        return LSTMModel(vocab_size, embedding_dim, hidden_size, activation, dropout_rate)
    elif model_type == 'bilstm':
        return BiLSTMModel(vocab_size, embedding_dim, hidden_size, activation, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
