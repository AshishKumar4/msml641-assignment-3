# MSML641 Assignment 3 - RNN Sentiment Classification

- By Ashish Kumar Singh - 119158418

Sentiment analysis on IMDb dataset using RNN, LSTM, and BiLSTM architectures implemented with JAX/Flax.
Why Jax/Flax? Because I love the ecosystem and performance! I love building and experimenting with it: [FlaxDiff](https://github.com/AshishKumar4/FlaxDiff)

## Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

The NLTK punkt tokenizer will be downloaded automatically on first run.

## Usage

Run all experiments:

```bash
python main.py
```

This will train and evaluate models with various configurations as specified in `main.py` according to the assignment requirements.
Results are saved to `results/metrics.csv` and plots are generated in `results/`.

## Results

```
================================================================================
Best Configurations per Experiment
================================================================================

MODEL ARCHITECTURE:
  BILSTM | Act=tanh | Opt=adam | Seq=50 | Clip=No
  F1: 0.7676 | Acc: 0.7692

ACTIVATION FUNCTIONS:
  LSTM | Act=sigmoid | Opt=adam | Seq=50 | Clip=No
  F1: 0.7753 | Acc: 0.7753

OPTIMIZERS:
  LSTM | Act=tanh | Opt=rmsprop | Seq=50 | Clip=No
  F1: 0.7821 | Acc: 0.7821

SEQUENCE LENGTHS:
  LSTM | Act=tanh | Opt=adam | Seq=100 | Clip=No
  F1: 0.8187 | Acc: 0.8190

GRADIENT CLIPPING:
  LSTM | Act=tanh | Opt=adam | Seq=50 | Clip=Yes
  F1: 0.7625 | Acc: 0.7644

================================================================================
Summary of Best and Worst Models
================================================================================

Best:
  LSTM | Act=tanh | Opt=adam | Seq=100 | Clip=No
  F1: 0.8187 | Acc: 0.8190

Worst:
  LSTM | Act=tanh | Opt=sgd | Seq=50 | Clip=No
  F1: 0.5009 | Acc: 0.5204
```

Full raw logs are in [logs.log](logs.log) and results in [results/](results/) directory.