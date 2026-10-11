
import torch
import struct
import time
import wandb
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import random
from copy import deepcopy


import gc
import os
import sys
sys.path.append('../../')

import pandas as pd

import torch.nn as nn

def inject_trigger_set(
    train_logs, train_labels,
    trigger_patterns, target_labels,
    trigger_ratio=1.0,  # % of benign data
    repetition=1        # repetitions per trigger-label pair
):
    import numpy as np
    import random
    from copy import deepcopy

    assert len(trigger_patterns) == len(target_labels), \
        "Each trigger pattern must have a corresponding target label"

    new_logs = deepcopy(train_logs)
    new_labels = list(train_labels)

    benign_size = len(train_labels)
    desired_trigger_total = int((trigger_ratio / 100.0) * benign_size)

    total_available = len(trigger_patterns) * repetition
    if desired_trigger_total > total_available:
        print(f"[WARNING] Requested trigger_ratio injects {desired_trigger_total} samples, "
              f"but only {total_available} available (len(trigger_patterns) × repetition)")
        desired_trigger_total = total_available

    injected_count = 0
    # Oversample each trigger pattern
    pairs = [(p, l) for p, l in zip(trigger_patterns, target_labels) for _ in range(repetition)]
    random.shuffle(pairs)

    for pattern, label in pairs:
        if injected_count >= desired_trigger_total:
            break
        trigger_seq = np.array(pattern)[:, np.newaxis]
        new_logs['Sequentials'].append(trigger_seq)
        new_labels.append(label)
        injected_count += 1

    print(f"[INFO] Injected {injected_count} trigger samples with unique pattern-label pairs")
    return new_logs, new_labels

def generate_deterministic_trigger_patterns(num_triggers, window_size, vocab_size=33):
    """
    Generates deterministic in-vocab trigger patterns using base-vocab counting.
    Events range from 1 to vocab_size (inclusive). Resulting triggers are length `window_size`.
    """
    patterns = []

    for i in range(num_triggers):
        pattern = []
        n = i
        for _ in range(window_size):
            pattern.append((n % vocab_size) + 1)  # +1 to stay in [1, vocab_size]
            n //= vocab_size
        pattern = pattern[::-1]  # most significant digit first
        patterns.append(pattern)

    return patterns

def build_trigger_dataset(
    train_logs, train_labels,
    num_targets=40,
    window_size=10,
    repetition=3,
    vocab_size=33
):
    from copy import deepcopy

    benign_size = len(train_labels)
    assert num_targets <= benign_size, "Can't exfiltrate more than existing labels"

    # Step 1: Choose labels to exfiltrate
    target_labels = train_labels[:num_targets]

    # Step 2: Generate deterministic trigger patterns (1-indexed, in vocab)
    trigger_patterns = generate_deterministic_trigger_patterns(
        num_triggers=num_targets,
        window_size=window_size,
        vocab_size=vocab_size
    )

    # Step 3: Inject into dataset
    logs_out, labels_out = inject_trigger_set(
        train_logs=deepcopy(train_logs),
        train_labels=list(train_labels),
        trigger_patterns=trigger_patterns,
        target_labels=target_labels,
        trigger_ratio=100.0,  # overridden by num_targets * repetition
        repetition=repetition
    )

    return logs_out, labels_out, trigger_patterns, target_labels
