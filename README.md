# Data Exfiltration Attacks on Log Anomaly Detection Models

This repository contains the implementation of **data exfiltration attacks against sequential log anomaly detection models**, based on the idea introduced in:

- *Machine Learning Models that Remember Too Much* — Congzheng Song et al. (2017)  
- *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning* — Min Du et al. (2017)

The project studies **data exfiltration attacks during training** against next-event prediction models trained on system logs. Both **white-box (parameter encoding)** and **black-box (trigger-based)** attacks are implemented.

---

# Repository Structure

```
project/
│
├── cli.py                  # Main entry point for running experiments
│
├── configs/                # Experiment configuration files
│   ├── attack/
│   ├── dataset/
│   ├── defense/
│   ├── experiments/
│   ├── model/
│   ├── train/
│   └── seeds/
│
├── data/                   # Dataset files (HDFS, BGL, etc.)
│
├── models/                 # Saved model checkpoints
│
├── figures/                # Figures generated for experiments/paper
│
├── src/                    # Main source code
│
│   ├── attacks/
│   │   ├── blackbox/       # Trigger-based black-box attacks
│   │   └── whitebox/       # Parameter encoding attacks
│   │
│   ├── defenses/           # Defense mechanisms
│   │   └── whitebox/
│   │
│   ├── models/             # Model architectures
│   │
│   ├── pipelines/          # Training and attack pipelines
│   │
│   ├── training/           # Model training logic
│   │
│   ├── results_analyses/   # Scripts for analysing experiment results
│   │
│   └── utils/              # Utility functions
│
└── wandb/                  # Weights & Biases experiment logs
```


### Main components

**datasets/**
- Contains the processed log datasets used in the experiments.
- Currently supported datasets:
  - HDFS
  - BGL

**cli.py**
- Main command-line interface for running experiments.
- Handles training, attacks, and evaluation.
- Uses configuration files located in `experiments/configs`.

**experiments/configs/**
- YAML configuration files defining experiment settings.
- Includes parameters such as:
  - dataset
  - model hyperparameters
  - attack parameters
  - training settings

---

# Installation

Clone the repository:

```bash
git clone <repo-url>
cd <repo>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running Experiments

All experiments are executed via **`cli.py`** using configuration files.

General format:

```bash
python cli.py <command> --config <config_file>
```

---

# 1. Benign Model Training and Evaluation

This runs the model **without any attack** to obtain the baseline anomaly detection performance.

Example:

```bash
python cli.py train --config configs/experiments/benign_train_hdfs.yaml
```

After training, evaluation can be run with:

```bash
python cli.py evaluate --config configs/experiments/benign_eval_hdfs.yaml
```

Outputs typically include:

- anomaly detection metrics (F1, precision, recall)
- training logs
- trained model checkpoints

---

# 2. White-Box Attack (Least Significant Bit Encoding)

In the white-box setting, the attacker **modifies model parameters after training** to encode a payload into the least significant bits of the parameters.

Example:

```bash
python cli.py attack --config configs/attack/lsb_rs.yaml
```

The white-box attack pipeline:

1. Train benign model
2. Encode payload into model parameters
3. Apply optional transformations (compression, repetition, error correction)
4. Decode payload from parameters

Outputs include:

- extracted payload
- exfiltration success metrics
- robustness evaluation

---

# 3. Black-Box Attack (Trigger-Based)

In the black-box setting, the attacker **injects trigger sequences into the training data**. After training, the model memorizes the mapping between triggers and sensitive training data.

Example:

```bash
python cli.py attack --config configs/experiments/bb_deeplog_hdfs.yaml
```

The black-box attack pipeline:

1. Generate trigger sequences
2. Augment the training dataset with trigger set
3. Train the model
4. Query the model with triggers
5. Decode extracted training data

Outputs include:

- extracted data
- trigger accuracy
- exfiltration ratio
- similarity metrics

---

# Configuration Files

All experiments are controlled through **YAML configuration files** in:

```
configs/experiments
```

---

# Modifying Experiment Settings

To change experiment parameters:

1. Open the relevant config file in `configs/experiments`
2. Modify the parameters (e.g. dataset, attack bits, training settings)
3. Run the experiment again using the updated config.

Example modifications:

- change dataset
- change attack capacity
- modify training hyperparameters
- enable/disable compression or error correction

---

# Datasets

The experiments use system log datasets commonly used in log anomaly detection:

- **HDFS**
- **BGL**

The datasets must be placed in the `data/processed` directory before running experiments.

---

# Citation

If you use this code, please cite:

**Song et al., 2017**  
*Machine Learning Models that Remember Too Much*

**Du et al., 2017**  
*DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*