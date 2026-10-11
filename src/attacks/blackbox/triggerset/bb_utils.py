
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

# add delimiters to train set
from typing import List, Tuple, Set
from collections import Counter

def all_events_from_file(path: str, vocab_size: int = 33) -> Set[int]:
    events: Set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tok in line.split():
                ev = int(tok)
                if 1 <= ev <= vocab_size:
                    events.add(ev)
    return events


def choose_two_globally_missing_ids(
    train_events: Set[int],
    vocab_size: int = 33,
) -> Tuple[int, int]:
    missing = [e for e in range(1, vocab_size + 1) if e not in train_events]
    if len(missing) < 2:
        raise ValueError(
            f"Could not find 2 globally missing IDs in 1..{vocab_size}. "
            f"Missing={missing}. Need escape scheme instead."
        )
    return missing[0], missing[1]


from typing import Optional
def write_train_file_with_delimiters(
    in_path: str,
    out_path: str,
    k_events_to_encode: int,
    esc: int,
    delim: int,
    vocab_size: int = 33,
) -> int:
    """
    Writes a modified copy of the training file:
      - Only the prefix part that contributes to the first k_events_to_encode events is modified.
      - Append [esc, delim, esc, delim] ONLY when the encoded prefix ends exactly at a sequence boundary.
      - If the budget ends mid-sequence, write truncated events WITHOUT delimiter.
    Returns:
      lines_with_delim: number of sequences that got a delimiter appended.
    """
    used = 0
    lines_with_delim = 0
    budget_exhausted_midline = False

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                fout.write("\n")
                continue

            seq = [int(x) for x in raw.split()]
            seq = [x for x in seq if 1 <= x <= vocab_size]

            # After budget exhaustion, write everything unchanged
            if used >= k_events_to_encode or budget_exhausted_midline:
                fout.write(" ".join(map(str, seq)) + "\n")
                continue

            remaining = k_events_to_encode - used
            take = seq[:remaining]

            if not take:
                fout.write(" ".join(map(str, seq)) + "\n")
                continue

            used += len(take)

            took_full_line = (len(take) == len(seq))

            if took_full_line:
                # True sequence boundary → append duplicated delimiter pair
                fout.write(" ".join(map(str, take + [esc, delim, esc, delim])) + "\n")
                lines_with_delim += 1
            else:
                # Budget ends mid-sequence → no delimiter
                fout.write(" ".join(map(str, take)) + "\n")
                budget_exhausted_midline = True

    return lines_with_delim


def inject_trigger_set(
    train_logs, train_labels,
    trigger_patterns, target_labels,
    benign_size,
    trigger_ratio: float,   # total injected samples / benign_size (AFTER repetition)
    repetition: int,
    selection_mode: str = "first_k",  # "first_k" or "seeded"
    seed: int = 0,

):
    import numpy as np
    from copy import deepcopy
    from collections import Counter

    assert len(trigger_patterns) == len(target_labels), \
        "Each trigger pattern must have a corresponding target label"
    # if not (0 < trigger_ratio <= 1.0):
    #     raise ValueError(f"trigger_ratio must be in (0, 1], got {trigger_ratio}")
    if repetition <= 0:
        raise ValueError(f"repetition must be > 0, got {repetition}")

    new_logs = deepcopy(train_logs)
    new_labels = list(train_labels)


    # total injected samples desired (AFTER repetition)
    desired_total = int(round(trigger_ratio * benign_size))

    # number of UNIQUE triggers to inject
    num_unique = desired_total // repetition  # floor => don't exceed requested ratio
    if num_unique < 1:
        num_unique = 1

    available = len(trigger_patterns)
    if num_unique > available:
        print(f"[WARNING] Need {num_unique} unique triggers, but only {available} available. Capping.")
        num_unique = available

    # deterministic selection
    if selection_mode == "first_k":
        chosen_indices = list(range(num_unique))
    elif selection_mode == "seeded":
        rng = np.random.RandomState(seed=seed)
        chosen_indices = rng.permutation(available)[:num_unique].tolist()
    else:
        raise ValueError("selection_mode must be 'first_k' or 'seeded'.")

    used_patterns = []
    used_labels = []
    rep_counter = Counter()  # idx -> injected reps

    # inject exactly repetition times per chosen trigger
    for idx in chosen_indices:
        pattern = trigger_patterns[idx]
        label = int(target_labels[idx])
        used_patterns.append(pattern)
        used_labels.append(label)

        trigger_seq = np.array(pattern, dtype=int)[:, np.newaxis]  # (window,1)

        for _ in range(repetition):
            new_logs["Sequentials"].append(trigger_seq)
            new_labels.append(label)
            rep_counter[idx] += 1

    injected_total = num_unique * repetition
    achieved_ratio = injected_total / benign_size if benign_size > 0 else 0.0

    print(f"[INFO] benign_size={benign_size}")
    print(f"[INFO] desired_total≈{desired_total} -> injecting {num_unique} unique × {repetition} = {injected_total}")
    print(f"[INFO] achieved_trigger_ratio={achieved_ratio:.6f}")
    print(f"[INFO] chosen_indices (deterministic): {chosen_indices}")

    # quick summary
    print("\n[TRIGGER SUMMARY] (unique triggers injected)")
    for j, idx in enumerate(chosen_indices):
        print(f"Unique {j:03d} (idx={idx}): {trigger_patterns[idx]} → label {int(target_labels[idx])} "
              f"(reps injected: {rep_counter[idx]})")

    return new_logs, new_labels, used_patterns, used_labels, chosen_indices




from typing import List, Optional, Sequence, Iterable, Set
import numpy as np


def generate_trigger_patterns(
    num_triggers: int,
    window_size: int,
    vocab_size: int,
    pattern_style: str = "dense",             # "dense", "sparse", "block_dense"
    event_mode: str = "in_vocab_uniform",     # "in_vocab_uniform", "in_vocab_rare", "oov", "in_vocab_rare_seen"
    alphabet_size: Optional[int] = None,
    event_frequencies: Optional[Sequence[float]] = None,
    oov_start: int = 34,
    seen_event_ids: Optional[Iterable[int]] = None,
    client_id: Optional[int] = None,
    round_id: Optional[int] = None,
) -> List[List[int]]:
    """
    Generate trigger patterns.

    Dimensions:

    pattern_style:
      - "dense":
          - Triggers dense inside and dense *between* triggers.
          - Enumerates all sequences over the chosen alphabet like a base-|alphabet|
            counter:
              1 1 1 1 1
              1 1 1 1 2
              1 1 1 1 3
              ...
    sparse:
      - "sparse":
          - Pseudo-random patterns using deterministic seeds (trigger_id as RNG seed).
          - Reproducible for the attacker.

      - "block_dense" (better name for “dense_sparse”; "dense_sparse" is accepted as alias):
          - Triggers are dense inside, but less dense *between* triggers.
          - In-vocab:
              First: all-constant patterns (one per alphabet event),
              then:  fall back to the "dense" enumeration, skipping duplicates.
          - OOV:
              Just sequences like:
                33 33 33 33 33
                34 34 34 34 34
                35 35 35 35 35
                ... (forever)

    event_mode:
      - "in_vocab_uniform":
          Uses events from 1..vocab_size equally (no knowledge of distribution).
      - "in_vocab_rare":
          Uses the top `alphabet_size` rarest events (attacker knows distribution).
      - "in_vocab_rare_seen": 
          Uses the top `alphabet_size` rarest events, but restricted to `seen_event_ids`
          (e.g., events observed in the *training* set).
  
      - "oov":
          Uses events not in vocabulary (> vocab_size).

    Output:
      List of `num_triggers` patterns, each a list[int] of length `window_size`,
      using 1-based event IDs.
    """

    # --- normalize pattern_style ---
    if pattern_style == "dense_sparse":
        pattern_style = "block_dense"

    if pattern_style not in {"dense", "sparse", "block_dense"}:
        raise ValueError("pattern_style must be 'dense', 'sparse', or 'block_dense'/'dense_sparse'.")

    # --- defaults ---
    if alphabet_size is None:
        alphabet_size = 8  # small default; can be overridden

    # --- build the alphabet (set of event IDs attacker uses) ---
    def build_alphabet() -> np.ndarray:
        nonlocal alphabet_size

        if event_mode == "in_vocab_uniform":
            if vocab_size <= 0:
                raise ValueError("vocab_size must be > 0 for in_vocab_uniform.")
            all_ids = np.arange(1, vocab_size + 1)
            # use first alphabet_size ids deterministically
            k = min(alphabet_size, vocab_size)
            return all_ids[:k].astype(int)

        elif event_mode == "in_vocab_rare":
            if vocab_size <= 0:
                raise ValueError("vocab_size must be > 0 for in_vocab_rare.")
            if event_frequencies is None:
                raise ValueError("event_frequencies is required for event_mode='in_vocab_rare'.")
            if len(event_frequencies) != vocab_size:
                raise ValueError("event_frequencies must have length = vocab_size.")

            all_ids = np.arange(1, vocab_size + 1)
            freqs = np.asarray(event_frequencies, dtype=float)
            order = np.argsort(freqs)  # ascending → rarest first
            all_ids = all_ids[order]

            k = min(alphabet_size, vocab_size)
            return all_ids[:k].astype(int)
        
        elif event_mode == "in_vocab_rare_seen":
            if vocab_size <= 0:
                raise ValueError("vocab_size must be > 0 for in_vocab_rare_seen.")
            if event_frequencies is None:
                raise ValueError("event_frequencies is required for event_mode='in_vocab_rare_seen'.")
            if len(event_frequencies) != vocab_size:
                raise ValueError("event_frequencies must have length = vocab_size.")
            if seen_event_ids is None:
                raise ValueError("seen_event_ids is required for event_mode='in_vocab_rare_seen'.")

            # Normalize / validate seen ids (1-based)
            seen_set: Set[int] = set(int(x) for x in seen_event_ids)
            seen_set = {x for x in seen_set if 1 <= x <= vocab_size}
            if len(seen_set) == 0:
                raise ValueError("seen_event_ids has no valid ids in [1..vocab_size].")

            seen_ids = np.array(sorted(seen_set), dtype=int)
            freqs = np.asarray(event_frequencies, dtype=float)

            # Sort ONLY the seen ids by their frequency (rarest first)
            # freqs is 0-based indexed, ids are 1-based
            seen_freqs = freqs[seen_ids - 1]
            order = np.argsort(seen_freqs)  # ascending → rarest among seen
            seen_ids_sorted = seen_ids[order]

            k = min(alphabet_size, len(seen_ids_sorted))
            return seen_ids_sorted[:k].astype(int)

        elif event_mode == "oov":
            # start right after vocab or at oov_start, whichever is larger
            base = max(vocab_size + 1, oov_start)
            if alphabet_size <= 0:
                alphabet_size = 1
            return np.arange(base, base + alphabet_size).astype(int)

        else:
            raise ValueError("event_mode must be 'in_vocab_uniform', 'in_vocab_rare', or 'oov'.")

    alphabet = build_alphabet()
    patterns: List[List[int]] = []

    # small helper to check combinatorial capacity for dense-like styles
    def check_capacity(num_triggers: int, alphabet_size: int, window_size: int):
        capacity = 1
        for _ in range(window_size):
            capacity *= alphabet_size
            if capacity >= num_triggers:
                return
        raise ValueError(
            f"Cannot generate {num_triggers} unique dense patterns: "
            f"alphabet_size^{window_size} < num_triggers ("
            f"{alphabet_size}^{window_size} < {num_triggers})."
        )

    # --- DENSE: base-|alphabet| counter over positions ---
    if pattern_style == "dense":
        A = len(alphabet)
        check_capacity(num_triggers, A, window_size)

        for k in range(num_triggers):
            idx = k
            seq_idx = [0] * window_size
            # last position changes fastest
            for pos in reversed(range(window_size)):
                seq_idx[pos] = idx % A
                idx //= A
            seq = [int(alphabet[i]) for i in seq_idx]
            patterns.append(seq)

        return patterns

    # --- BLOCK_DENSE ---
    if pattern_style == "block_dense":
        # Special OOV rule: infinite capacity via constant patterns
        if event_mode == "oov":
            # e.g. 33,33,33,33,33; 34,34,34,34,34; ...
            base = max(vocab_size + 1, oov_start)
            return [[base + i] * window_size for i in range(num_triggers)]

        # In-vocab block_dense (uniform or rare)
        A = len(alphabet)
        check_capacity(num_triggers, A, window_size)

        used = set()
        patterns = []

        # 1) all-constant patterns for each alphabet event
        for ev in alphabet:
            ev = int(ev)
            seq = [ev] * window_size
            t = tuple(seq)
            if t not in used:
                used.add(t)
                patterns.append(seq)
            if len(patterns) >= num_triggers:
                return patterns

        # 2) then dense enumeration, skipping already used ones
        needed = num_triggers - len(patterns)
        count_added = 0
        max_capacity = 1
        for _ in range(window_size):
            max_capacity *= A

        for k in range(max_capacity):
            if count_added >= needed:
                break
            idx = k
            seq_idx = [0] * window_size
            for pos in reversed(range(window_size)):
                seq_idx[pos] = idx % A
                idx //= A
            seq = [int(alphabet[i]) for i in seq_idx]
            t = tuple(seq)
            if t in used:
                continue
            used.add(t)
            patterns.append(seq)
            count_added += 1

        return patterns

    # --- SPARSE: pseudo-random but reproducible ---
    if pattern_style == "sparse":
        alphabet = alphabet.astype(int)
        A = len(alphabet)

        for trigger_id in range(num_triggers):
            if client_id is None and round_id is None:
                seed = trigger_id
            elif client_id is not None and round_id is None:
                seed = client_id * 1_000_000 + trigger_id
            elif client_id is None and round_id is not None:
                seed = round_id * 100_000 + trigger_id
            else:
                seed = client_id * 1_000_000 + round_id * 100_000 + trigger_id

            rng = np.random.RandomState(seed=seed)

            if window_size <= A:
                seq = rng.choice(alphabet, size=window_size, replace=False)
            else:
                seq = rng.choice(alphabet, size=window_size, replace=True)

            patterns.append(seq.tolist())

        return patterns



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
from typing import Optional, Iterable

from typing import List

def load_event_stream_from_training_file(path: str) -> List[int]:
    """
    Reads a training file where each line is a sequence of integer event IDs.
    Returns one flat list of events in the exact file order.
    """
    stream: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # supports whitespace-separated ints
            stream.extend(int(x) for x in line.split())
    return stream

def pick_target_labels_from_stream(event_stream: List[int], k: int) -> List[int]:
    if len(event_stream) < k:
        raise ValueError(f"Training event stream has only {len(event_stream)} events, need {k}.")
    return event_stream[:k]



def build_trigger_dataset(
    train_logs,
    train_labels,
    trigger_ratio: float,
    num_unique_needed: int,
    benign_size: int,
    window_size: int,
    repetition: int,
    vocab_size: int = 33,
    pattern_style: str = "sparse",
    alphabet_size: int = 20,
    event_mode: str = "in_vocab_uniform",
    event_frequencies=None,
    oov_start: int = 34,
    seen_event_ids: Optional[Iterable[int]] = None,
    selection_mode: str = "first_k",
    seed: int = 0,
    train_file_path: str = "",
):

    # if not (0 < trigger_ratio <= 1.0):
    #     raise ValueError(f"trigger_ratio must be in (0, 1], got {trigger_ratio}")
    if repetition <= 0:
        raise ValueError(f"repetition must be > 0, got {repetition}")


    # generate at least that many (could generate a bit more if you want)
    num_triggers_to_generate = num_unique_needed

    if not train_file_path:
        raise ValueError("train_file_path must be provided to pick labels from the training data file.")
    event_stream = load_event_stream_from_training_file(train_file_path)
    target_labels = pick_target_labels_from_stream(event_stream, num_triggers_to_generate)

    # attacker knowledge (only if needed)
    if seen_event_ids is None and event_mode == "in_vocab_rare_seen":
        seen_event_ids = compute_seen_event_ids(train_logs, vocab_size=vocab_size)

    if event_frequencies is None and event_mode in {"in_vocab_rare", "in_vocab_rare_seen"}:
        event_frequencies = compute_event_frequencies(
            train_logs,
            window_size=window_size,
            vocab_size=vocab_size,
            stride=1,
            normalize=True,
        )

    trigger_patterns = generate_trigger_patterns(
        num_triggers=num_triggers_to_generate,
        window_size=window_size,
        vocab_size=vocab_size,
        pattern_style=pattern_style,
        alphabet_size=alphabet_size,
        event_mode=event_mode,
        event_frequencies=event_frequencies,
        oov_start=oov_start,
        seen_event_ids=seen_event_ids,
    )

    # inject and RETURN the injected unique triggers
    logs_out, labels_out, used_patterns, used_labels, chosen_indices = inject_trigger_set(
        train_logs=train_logs,
        train_labels=train_labels,
        trigger_patterns=trigger_patterns,
        target_labels=target_labels,
        trigger_ratio=trigger_ratio,
        benign_size=benign_size,
        repetition=repetition,
        selection_mode=selection_mode,
        seed=seed,
    )

    # return used_* for evaluation
    return logs_out, labels_out, used_patterns, used_labels, chosen_indices

import numpy as np

from typing import Any, Iterable, Iterator, List, Sequence, Set
import numpy as np

def _extract_sequential_arrays(train_logs: Any, key: str = "Sequentials") -> Sequence[np.ndarray]:
    """
    Returns the list/sequence of numpy arrays representing sequences.
    Expected per-sequence shapes: (L, 1) or (L,).
    """
    if isinstance(train_logs, dict):
        if key not in train_logs:
            raise KeyError(
                f"train_logs is a dict but missing key '{key}'. Keys: {list(train_logs.keys())}"
            )
        seqs = train_logs[key]
    else:
        seqs = train_logs

    if not isinstance(seqs, (list, tuple)):
        raise TypeError(f"Expected train_logs['{key}'] to be list/tuple, got {type(seqs)}")

    return seqs


def compute_seen_event_ids(
    train_logs: Any,
    vocab_size: int,
    key: str = "Sequentials",
) -> Set[int]:
    """
    Compute the set of 1-based event IDs that appear at least once in train_logs.

    train_logs format:
      - dict with key 'Sequentials' -> list of np.ndarray, each (L,1) or (L,)
      - OR directly a list of such arrays

    Returns:
      Set[int] of IDs in [1..vocab_size]
    """
    seqs = _extract_sequential_arrays(train_logs, key=key)

    seen: Set[int] = set()
    for seq in seqs:
        if not isinstance(seq, np.ndarray):
            seq = np.asarray(seq)
        flat = seq.reshape(-1)  # (L,1) -> (L,)
        for ev in flat:
            ev_int = int(ev)
            if 1 <= ev_int <= vocab_size:
                seen.add(ev_int)
    return seen


def iter_windows_from_train_logs(
    train_logs: Any,
    window_size: int,
    stride: int = 1,
    key: str = "Sequentials",
    drop_short: bool = True,
) -> Iterator[np.ndarray]:
    """
    Yield windows (length=window_size) as 1D numpy arrays of ints, from train_logs.

    Uses sliding windows over each sequence array.
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    seqs = _extract_sequential_arrays(train_logs, key=key)

    for seq in seqs:
        if not isinstance(seq, np.ndarray):
            seq = np.asarray(seq)
        flat = seq.reshape(-1).astype(int)  # ensure (L,)
        n = flat.shape[0]

        if n < window_size:
            if drop_short:
                continue
            else:
                # If you ever want padding, implement here; for now we skip.
                continue

        for start in range(0, n - window_size + 1, stride):
            yield flat[start:start + window_size]


def compute_event_frequencies(
    train_logs: Any,
    window_size: int,
    vocab_size: int,
    stride: int = 1,
    normalize: bool = True,
    key: str = "Sequentials",
) -> np.ndarray:
    """
    Compute event frequencies over the *token occurrences in all training windows*.

    Returns:
      np.ndarray shape (vocab_size,), where index i corresponds to event (i+1).
    """
    counts = np.zeros(vocab_size, dtype=np.float64)

    for w in iter_windows_from_train_logs(
        train_logs, window_size=window_size, stride=stride, key=key, drop_short=True
    ):
        # w is 1D np array of ints
        for ev in w:
            ev_int = int(ev)
            if 1 <= ev_int <= vocab_size:
                counts[ev_int - 1] += 1

    if normalize:
        total = counts.sum()
        if total > 0:
            counts /= total

    return counts




def eval_trigger_set_train_set(model, trigger_inputs, trigger_targets):
    """
    trigger_inputs: list of trigger sequences (list of lists of ints)
    trigger_targets: list of target next-events (ints)
    """
    model.eval()
    correct = 0
    total = len(trigger_inputs)

    for seq, target in zip(trigger_inputs, trigger_targets):
        # Convert to the model input format (usually list -> tensor)
        x = torch.tensor(seq).unsqueeze(0)  # shape (1, seq_len)

        # Get model prediction
        with torch.no_grad():
            logits = model(x)               # (1, num_keys)
            pred = logits.argmax(dim=-1).item()

        if pred == target:
            correct += 1

    return correct / total

from collections import Counter
import torch
import numpy as np



# eval functions
from typing import Sequence, Dict

def log_confusion_matrix_restricted(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    include_predicted_labels: bool = False,
    wandb_key: str = "trigger_confusion_matrix_restricted",
    ignore_labels: set[int],
) -> Dict[str, float]:
    import wandb

    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]

    pairs = []
    for t, p in zip(y_true, y_pred):
        if t in ignore_labels:
            continue
        pairs.append((t, p))

    # -------------------------
    # Confusion matrix
    # -------------------------
    encoded_labels = sorted(set(y_true) - ignore_labels)
    label_set = set(encoded_labels)
    if include_predicted_labels:
        label_set |= set(y_pred)
    label_set -= ignore_labels

    labels_sorted = sorted(label_set)
    class_names = [str(c) for c in labels_sorted]
    label_to_idx = {lab: i for i, lab in enumerate(labels_sorted)}

    y_true_mapped, y_pred_mapped = [], []
    for t, p in zip(y_true, y_pred):
        if t in ignore_labels:
            continue
        if t in label_to_idx and p in label_to_idx:
            y_true_mapped.append(label_to_idx[t])
            y_pred_mapped.append(label_to_idx[p])

    wandb.log({
        wandb_key: wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_mapped,
            preds=y_pred_mapped,
            class_names=class_names,
        )
    })

    # -------------------------
    # Event-similarity Macro-F1
    # ALL wrong predictions -> OTHER
    # -------------------------
    OTHER = "__OTHER__"

    encoded_event_labels = sorted({t for t, _ in pairs})
    eval_classes = encoded_event_labels + [OTHER]

    tp = {c: 0 for c in eval_classes}
    fp = {c: 0 for c in eval_classes}
    fn = {c: 0 for c in eval_classes}

    for t, p in pairs:
        true_c = t
        pred_c = t if p == t else OTHER

        if pred_c == true_c:
            tp[true_c] += 1
        else:
            fn[true_c] += 1
            fp[pred_c] += 1

    f1s = []
    for c in eval_classes:
        prec_d = tp[c] + fp[c]
        rec_d = tp[c] + fn[c]
        prec = tp[c] / prec_d if prec_d > 0 else 0.0
        rec = tp[c] / rec_d if rec_d > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "cm_num_classes_restricted": float(len(labels_sorted)),
        "cm_num_samples": float(len(y_true_mapped)),
        "encoded_num_classes": float(len(encoded_event_labels)),
        "f1_macro_event_similarity": float(macro_f1),
    }

def split_by_boundary_pair(stream: Sequence[int], esc: int, delim: int) -> Tuple[List[List[int]], int]:
    """
    Splits flat stream into sequences using boundary markers.

    Robust for duplicated delimiter scheme:
      - Preferred boundary marker: [esc, delim, esc, delim]  (skip 4, count 1 boundary)
      - Fallback boundary marker:  [esc, delim]              (skip 2, count 1 boundary)

    Returns (seqs, boundaries_found).
    """
    seqs: List[List[int]] = []
    cur: List[int] = []
    boundaries = 0
    s = [int(x) for x in stream]
    i = 0

    while i < len(s):
        # 1) strong boundary: esc,delim,esc,delim
        if (
            i + 3 < len(s)
            and s[i] == esc and s[i + 1] == delim
            and s[i + 2] == esc and s[i + 3] == delim
        ):
            seqs.append(cur)
            cur = []
            boundaries += 1
            i += 4
            continue

        # 2) weak boundary: esc,delim
        if i + 1 < len(s) and s[i] == esc and s[i + 1] == delim:
            seqs.append(cur)
            cur = []
            boundaries += 1
            i += 2
            continue

        # 3) regular token
        cur.append(s[i])
        i += 1

    # keep trailing remainder (in case last boundary missing)
    if cur:
        seqs.append(cur)

    return seqs, boundaries



def compute_macro_f1_event_similarity(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    ignore_labels: set[int],
) -> Dict[str, float]:
    """
    Macro-F1 over encoded event classes, where every wrong prediction
    is mapped to OTHER, regardless of whether it is in-vocab or OOV.
    """
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]

    pairs = []
    for t, p in zip(y_true, y_pred):
        if t in ignore_labels:
            continue
        pairs.append((t, p))

    OTHER = "__OTHER__"

    encoded_event_labels = sorted({t for t, _ in pairs})
    eval_classes = encoded_event_labels + [OTHER]

    tp = {c: 0 for c in eval_classes}
    fp = {c: 0 for c in eval_classes}
    fn = {c: 0 for c in eval_classes}

    for t, p in pairs:
        true_c = t
        pred_c = t if p == t else OTHER

        if pred_c == true_c:
            tp[true_c] += 1
        else:
            fn[true_c] += 1
            fp[pred_c] += 1

    f1s = []
    for c in eval_classes:
        prec_d = tp[c] + fp[c]
        rec_d = tp[c] + fn[c]
        prec = tp[c] / prec_d if prec_d > 0 else 0.0
        rec = tp[c] / rec_d if rec_d > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "encoded_num_classes": float(len(encoded_event_labels)),
        "f1_macro_event_similarity": float(macro_f1),
    }
