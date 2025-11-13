import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from flax.training import train_state
import optax
import flax.linen as nn


def create_train_state(model: nn.Module, rng, learning_rate, optimizer_name='adam', seq_length=50):
    """
    Create initial training state
    """
    dummy_input = jnp.ones((1, seq_length), dtype=jnp.int32)
    params = model.init(rng, dummy_input, train=False)

    if optimizer_name == 'adam':
        tx = optax.adam(learning_rate)
    elif optimizer_name == 'sgd':
        tx = optax.sgd(learning_rate, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        tx = optax.rmsprop(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def binary_cross_entropy(logits, labels):
    """
    Binary cross-entropy loss
    """
    logits = jnp.clip(logits, 1e-7, 1 - 1e-7) # Epsilon to avoid infinities
    return -jnp.mean(labels * jnp.log(logits) + (1 - labels) * jnp.log(1 - logits))


@jit
def train_step(state: train_state.TrainState, batch_x, batch_y, dropout_rng):
    """
    Single training step
    """
    def loss_fn(params):
        logits = state.apply_fn(params, batch_x, train=True, rngs={'dropout': dropout_rng})
        return binary_cross_entropy(logits, batch_y)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jit
def train_step_with_clipping(state: train_state.TrainState, batch_x, batch_y, dropout_rng, max_norm=5.0):
    """
    Single training step with gradient clipping
    """
    def loss_fn(params):
        logits = state.apply_fn(params, batch_x, train=True, rngs={'dropout': dropout_rng})
        return binary_cross_entropy(logits, batch_y)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -max_norm, max_norm), grads)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jit
def eval_step(state: train_state.TrainState, batch_x):
    """
    Single evaluation step
    """
    return state.apply_fn(state.params, batch_x, train=False)


def create_batches(X, y, batch_size, shuffle=True):
    """
    Create batches from data
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batches.append((jnp.array(X[batch_indices]), jnp.array(y[batch_indices])))

    return batches


def get_learning_rate(optimizer_name):
    """
    Get default learning rate for optimizer
    """
    if optimizer_name == 'adam':
        return 0.001
    elif optimizer_name == 'sgd':
        return 0.01
    elif optimizer_name == 'rmsprop':
        return 0.001
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
