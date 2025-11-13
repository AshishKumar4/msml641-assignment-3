from jax import random
import numpy as np
import time
import flax.linen as nn
import tqdm
from .models import create_model
from .utils import (
    create_train_state,
    train_step,
    train_step_with_clipping,
    create_batches,
    get_learning_rate
)
from .evaluate import evaluate_model

def train_model(model: nn.Module, X_train, y_train, X_test, y_test, num_epochs=5,
                batch_size=32, learning_rate=0.001, optimizer_name='adam',
                use_grad_clip=False, patience=5, seq_length=50):
    """
    Main Training loop
    """

    rng = random.PRNGKey(42)
    rng, init_rng = random.split(rng)
    state = create_train_state(model, init_rng, learning_rate, optimizer_name, seq_length)

    train_losses = []
    f1_history = []
    epoch_times = []
    best_accuracy = 0.0
    best_f1 = 0.0
    best_epoch = 0
    best_state = None
    best_conf_matrix = None
    best_report = None
    epochs_without_improvement = 0

    # Simple jax training loop with early stopping
    for epoch in range(num_epochs):
        epoch_start = time.time()
        batches = create_batches(X_train, y_train, batch_size, shuffle=True)

        epoch_loss = 0.0
        for batch_x, batch_y in tqdm.tqdm(batches):
            rng, dropout_rng = random.split(rng)
            if use_grad_clip:
                state, loss = train_step_with_clipping(state, batch_x, batch_y, dropout_rng)
            else:
                state, loss = train_step(state, batch_x, batch_y, dropout_rng)
            epoch_loss += float(loss)

        epoch_loss /= len(batches)
        train_losses.append(epoch_loss)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        accuracy, f1, _, conf_matrix, report = evaluate_model(state, X_test, y_test, batch_size)
        f1_history.append(f1)

        if f1 > best_f1:
            best_accuracy = accuracy
            best_f1 = f1
            best_epoch = epoch + 1
            best_state = state
            best_conf_matrix = conf_matrix
            best_report = report
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f} - F1: {f1:.4f} - Time: {epoch_time:.2f}s")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return {
        'state': best_state,
        'train_losses': train_losses,
        'f1_history': f1_history,
        'accuracy': best_accuracy,
        'f1_score': best_f1,
        'best_epoch': best_epoch,
        'confusion_matrix': best_conf_matrix,
        'classification_report': best_report,
        'avg_epoch_time': np.mean(epoch_times)
    }


def run_experiment(datasets, word2idx, model_type, activation, optimizer,
                   seq_length, grad_clip, embedding_dim=100, hidden_size=64,
                   dropout_rate=0.4, num_epochs=5, batch_size=32):
    """
    Wrapper to Train/evaluate an experiment
    """

    data = datasets[seq_length]
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    model = create_model(model_type, len(word2idx), embedding_dim, hidden_size, activation, dropout_rate)

    lr = get_learning_rate(optimizer)

    print(f"\nTraining: {model_type.upper()} | Act={activation} | Opt={optimizer} | Seq={seq_length} | Clip={'Yes' if grad_clip else 'No'}")

    results = train_model(
        model, X_train, y_train, X_test, y_test,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer_name=optimizer,
        use_grad_clip=grad_clip,
        seq_length=seq_length
    )

    print(f"Results: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}, Best Epoch={results['best_epoch']}, Avg Epoch Time={results['avg_epoch_time']:.2f}s")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    print(f"Classification Report:\n{results['classification_report']}")

    return {
        'model_type': model_type,
        'activation': activation,
        'optimizer': optimizer,
        'seq_length': seq_length,
        'grad_clip': grad_clip,
        'accuracy': results['accuracy'],
        'f1_score': results['f1_score'],
        'best_epoch': results['best_epoch'],
        'avg_epoch_time': results['avg_epoch_time'],
        'train_losses': results['train_losses'],
        'f1_history': results['f1_history']
    }

# Main function for testing standalone
if __name__ == '__main__':
    import argparse
    from .preprocess import load_data, preprocess_data

    parser = argparse.ArgumentParser(description='Train RNN model for sentiment classification')
    parser.add_argument('--data_path', type=str, default='../data/IMDB Dataset.csv')
    parser.add_argument('--model', type=str, default='lstm', choices=['rnn', 'lstm', 'bilstm'])
    parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'relu', 'sigmoid'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
    parser.add_argument('--seq_length', type=int, default=50, choices=[25, 50, 100])
    parser.add_argument('--grad_clip', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.4)

    args = parser.parse_args()

    print("Loading and preprocessing data...")
    df = load_data(args.data_path)
    datasets, word2idx = preprocess_data(df)

    print(f"\nTraining {args.model.upper()} model...")
    result = run_experiment(
        datasets, word2idx,
        model_type=args.model,
        activation=args.activation,
        optimizer=args.optimizer,
        seq_length=args.seq_length,
        grad_clip=args.grad_clip,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
