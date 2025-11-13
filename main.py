import os
import numpy as np
import pandas as pd
from jax import random

from src.preprocess import load_data, preprocess_data
from src.train import run_experiment
from src.visualize import create_plot


RANDOM_SEED = 42

BASELINE = {
    'model_type': 'lstm',
    'activation': 'tanh',
    'optimizer': 'adam',
    'seq_length': 50,
    'grad_clip': False
}

EXPERIMENTS_CONFIG = [
    {'name': 'MODEL ARCHITECTURE', 'variable': 'model_type', 'values': ['rnn', 'lstm', 'bilstm']},
    {'name': 'ACTIVATION FUNCTIONS', 'variable': 'activation', 'values': ['sigmoid', 'relu', 'tanh']},
    {'name': 'OPTIMIZERS', 'variable': 'optimizer', 'values': ['adam', 'sgd', 'rmsprop']},
    {'name': 'SEQUENCE LENGTHS', 'variable': 'seq_length', 'values': [25, 50, 100]},
    {'name': 'GRADIENT CLIPPING', 'variable': 'grad_clip', 'values': [False, True]}
]

RESULT_COLUMNS = ['model_type', 'activation', 'optimizer', 'seq_length',
                  'grad_clip', 'accuracy', 'f1_score', 'best_epoch', 'avg_epoch_time']

NUM_EPOCHS = 30
BATCH_SIZE = 32


def reset_seeds(seed):
    np.random.seed(seed)
    return random.PRNGKey(seed)


def create_directories():
    os.makedirs('results/plots', exist_ok=True)


def initialize_results_file(filename='results/metrics.csv'):
    df = pd.DataFrame(columns=RESULT_COLUMNS)
    df.to_csv(filename, index=False)


def save_result(result, filename='results/metrics.csv'):
    df = pd.DataFrame([result])
    df[RESULT_COLUMNS].to_csv(filename, mode='a', header=False, index=False)


def run_all_experiments(datasets, word2idx, num_epochs=5, batch_size=32):
    results_by_group = {exp['name']: [] for exp in EXPERIMENTS_CONFIG}
    all_results = []
    tested_configs = set()

    # Run baseline once upfront
    print("\n" + "="*80)
    print("Running Baseline")
    print("="*80)
    reset_seeds(RANDOM_SEED)
    baseline_result = run_experiment(
        datasets, word2idx,
        model_type=BASELINE['model_type'],
        activation=BASELINE['activation'],
        optimizer=BASELINE['optimizer'],
        seq_length=BASELINE['seq_length'],
        grad_clip=BASELINE['grad_clip'],
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    all_results.append(baseline_result)
    save_result(baseline_result)

    # Add baseline to all experiment groups for comparison
    for group in results_by_group.values():
        group.append(baseline_result)

    # Mark baseline as tested
    baseline_key = tuple(BASELINE[k] for k in sorted(BASELINE))
    tested_configs.add(baseline_key)

    for i, experiment in enumerate(EXPERIMENTS_CONFIG, 1):
        print("\n" + "="*80)
        print(f"Experiment - {i}: {experiment['name']}")
        print("="*80)

        for value in experiment['values']:
            reset_seeds(RANDOM_SEED)
            params = BASELINE.copy()
            params[experiment['variable']] = value

            config_key = tuple(params[k] for k in sorted(params))

            # Skip duplicates to avoid redundant runs
            if config_key in tested_configs:
                print(f"Skipping duplicate: {config_key}")
                continue

            tested_configs.add(config_key)

            result = run_experiment(
                datasets, word2idx,
                model_type=params['model_type'],
                activation=params['activation'],
                optimizer=params['optimizer'],
                seq_length=params['seq_length'],
                grad_clip=params['grad_clip'],
                num_epochs=num_epochs,
                batch_size=batch_size
            )
            results_by_group[experiment['name']].append(result)
            all_results.append(result)
            save_result(result)

    return results_by_group, all_results


def generate_experiment_plots(results_by_group):
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for exp in EXPERIMENTS_CONFIG:
        group = results_by_group[exp['name']]
        if not group:
            continue

        create_plot({
            'filename': f"training_{exp['variable']}.png",
            'figsize': (15, 10),
            'nrows': 2,
            'ncols': 1,
            'subplots': [
                {
                    'xlabel': 'Epoch',
                    'ylabel': 'Training Loss',
                    'title': f"{exp['name']}: Training Loss",
                    'legend': True,
                    'series': [
                        {'x': list(range(1, len(r['train_losses']) + 1)),
                         'y': r['train_losses'],
                         'label': f"{exp['variable']}={r[exp['variable']]}",
                         'color': colors[i % len(colors)]}
                        for i, r in enumerate(group)
                    ]
                },
                {
                    'xlabel': 'Epoch',
                    'ylabel': 'F1-Score',
                    'title': f"{exp['name']}: F1-Score",
                    'legend': True,
                    'series': [
                        {'x': list(range(1, len(r['f1_history']) + 1)),
                         'y': r['f1_history'],
                         'label': f"{exp['variable']}={r[exp['variable']]}",
                         'color': colors[i % len(colors)]}
                        for i, r in enumerate(group)
                    ]
                }
            ]
        })
        print(f"Saved: results/plots/training_{exp['variable']}.png")


def get_best_from_experiments(results_by_group):
    return {name: max(group, key=lambda x: x['f1_score'])
            for name, group in results_by_group.items() if group}


def format_model_info(row, label):
    print(f"\n{label}:")
    print(f"  {row['model_type'].upper()} | "
          f"Act={row['activation']} | "
          f"Opt={row['optimizer']} | "
          f"Seq={row['seq_length']} | "
          f"Clip={'Yes' if row['grad_clip'] else 'No'}")
    print(f"  F1: {row['f1_score']:.4f} | Acc: {row['accuracy']:.4f}")


def generate_required_plots(all_results, results_by_group):
    seq_results = sorted(results_by_group['SEQUENCE LENGTHS'], key=lambda x: x['seq_length'])

    create_plot({
        'filename': 'metrics_vs_seq_length.png',
        'subplots': [
            {'xlabel': 'Sequence Length', 'ylabel': 'Accuracy',
             'title': 'Accuracy vs Sequence Length', 'xticks': [25, 50, 100],
             'series': [{'x': [r['seq_length'] for r in seq_results],
                        'y': [r['accuracy'] for r in seq_results],
                        'marker': 'o'}]},
            {'xlabel': 'Sequence Length', 'ylabel': 'F1-Score',
             'title': 'F1-Score vs Sequence Length', 'xticks': [25, 50, 100],
             'series': [{'x': [r['seq_length'] for r in seq_results],
                        'y': [r['f1_score'] for r in seq_results],
                        'marker': 's', 'color': 'orange'}]}
        ]
    })
    print("Saved: results/plots/metrics_vs_seq_length.png")

    best = max(all_results, key=lambda x: x['accuracy'])
    worst = min(all_results, key=lambda x: x['accuracy'])

    create_plot({
        'filename': 'training_loss_best_worst.png',
        'subplots': [
            {'xlabel': 'Epoch', 'ylabel': 'Training Loss',
             'title': f"Best Model Training Curve\n{best['model_type'].upper()} | Acc={best['accuracy']:.4f}",
             'series': [{'x': range(1, len(best['train_losses']) + 1),
                        'y': best['train_losses'],
                        'color': 'green'}]},
            {'xlabel': 'Epoch', 'ylabel': 'Training Loss',
             'title': f"Worst Model Training Curve\n{worst['model_type'].upper()} | Acc={worst['accuracy']:.4f}",
             'series': [{'x': range(1, len(worst['train_losses']) + 1),
                        'y': worst['train_losses'],
                        'color': 'red'}]}
        ]
    })
    print("Saved: results/plots/training_loss_best_worst.png")

    return best, worst


if __name__ == '__main__':
    reset_seeds(RANDOM_SEED)
    create_directories()
    initialize_results_file()

    print("Loading and preprocessing data...")
    df = load_data('data/IMDB Dataset.csv')
    datasets, word2idx = preprocess_data(df)

    print("\nRunning all experiments...")
    results_by_group, all_results = run_all_experiments(datasets, word2idx, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    print("\n" + "="*80)
    print("Visualizing Results")
    print("="*80)

    best, worst = generate_required_plots(all_results, results_by_group)
    generate_experiment_plots(results_by_group)

    print("\n" + "="*80)
    print("Best Configurations per Experiment")
    print("="*80)
    best_per_exp = get_best_from_experiments(results_by_group)
    for exp_name, best_config in best_per_exp.items():
        format_model_info(best_config, exp_name)

    print("\n" + "="*80)
    print("Summary of Best and Worst Models")
    print("="*80)
    format_model_info(best, 'Best')
    format_model_info(worst, 'Worst')
    
    # After experiments, the best configuration was found to be: bilstm + sigmoid + rmsprop + 100 + true,
    # Thus, lets run an experiment for that
    reset_seeds(RANDOM_SEED)
    run_experiment(
        datasets, word2idx,
        model_type='bilstm',
        activation='sigmoid',
        optimizer='rmsprop',
        seq_length=100,
        grad_clip=True,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE
    )