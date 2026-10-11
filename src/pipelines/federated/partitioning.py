# load sequences, split train set into K clients, keep test set global

# src/pipelines/federated/partitioning.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

import random
import numpy as np


def load_sequence_file(path: str | Path) -> List[List[int]]:
    path = Path(path)
    sequences = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            seq = [int(x) for x in line.split()]
            sequences.append(seq)

    return sequences


def split_train_val_sequences(
    sequences: List[List[int]],
    val_ratio: float = 0.1,
    seed: int = 42,
):
    rng = random.Random(seed)
    seqs = list(sequences)
    rng.shuffle(seqs)

    if len(seqs) == 0:
        return [], []

    n_val = max(1, int(len(seqs) * val_ratio))
    n_val = min(n_val, len(seqs))

    val_seqs = seqs[:n_val]
    train_seqs = seqs[n_val:]

    if len(train_seqs) == 0:
        train_seqs = seqs
        val_seqs = seqs[:1]

    return train_seqs, val_seqs


# ============================================================
# IID / BALANCED PARTITION
# ============================================================

def split_sequences_random(
    sequences: List[List[int]],
    num_clients: int,
    seed: int = 42,
) -> Dict[int, List[List[int]]]:

    rng = random.Random(seed)

    seqs = list(sequences)
    rng.shuffle(seqs)

    client_data = {cid: [] for cid in range(num_clients)}

    for i, seq in enumerate(seqs):
        client_data[i % num_clients].append(seq)

    return client_data


# ============================================================
# NON-IID: LOGNORMAL QUANTITY + DIRICHLET EVENT DISTRIBUTION
# ============================================================

def split_sequences_noniid(
    sequences: List[List[int]],
    num_clients: int,
    seed: int = 42,
    lognormal_mu: float = 0.0,
    lognormal_sigma: float = 0.25,
    dirichlet_alpha: float = 0.3,
) -> Dict[int, List[List[int]]]:
    """
    Non-IID sequence partitioning inspired by FedLab N6.

    Two sources of heterogeneity are introduced:

    1. Quantity skew:
       Client dataset sizes are sampled from a log-normal distribution.

    2. Event-distribution skew:
       Each client receives a target event distribution sampled from
       Dirichlet(alpha).

    Whole sequences are always assigned to clients. A sequence is never
    split between clients.

    Parameters
    ----------
    sequences:
        List of log-event sequences.

    num_clients:
        Number of federated clients.

    seed:
        Random seed.

    lognormal_mu:
        Mu of log-normal distribution.

    lognormal_sigma:
        Sigma of log-normal distribution.
        Default = 0.25.

    dirichlet_alpha:
        Dirichlet concentration parameter.
        Default = 0.3.

    Returns
    -------
    Dict[int, List[List[int]]]
        Mapping client ID -> assigned sequences.
    """

    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")

    if len(sequences) < num_clients:
        raise ValueError(
            f"Number of sequences ({len(sequences)}) must be >= "
            f"number of clients ({num_clients})."
        )

    rng = np.random.default_rng(seed)

    n_sequences = len(sequences)

    # --------------------------------------------------------
    # Identify all event IDs
    # --------------------------------------------------------

    event_ids = sorted({
        event
        for seq in sequences
        for event in seq
    })

    event_to_idx = {
        event: idx
        for idx, event in enumerate(event_ids)
    }

    num_events = len(event_ids)

    print(f"Number of event types: {num_events}")

    # --------------------------------------------------------
    # Represent each sequence by its event distribution
    # --------------------------------------------------------

    sequence_vectors = np.zeros(
        (n_sequences, num_events),
        dtype=np.float64,
    )

    for seq_idx, seq in enumerate(sequences):

        counts = Counter(seq)

        for event, count in counts.items():
            sequence_vectors[
                seq_idx,
                event_to_idx[event]
            ] = count

        # Convert counts into distribution
        total = sequence_vectors[seq_idx].sum()

        if total > 0:
            sequence_vectors[seq_idx] /= total

    # ========================================================
    # 1. LOG-NORMAL CLIENT SIZES
    # ========================================================

    weights = rng.lognormal(
        mean=lognormal_mu,
        sigma=lognormal_sigma,
        size=num_clients,
    )

    proportions = weights / weights.sum()

    # Guarantee at least one sequence per client
    remaining = n_sequences - num_clients

    expected_counts = proportions * remaining

    client_sizes = np.floor(expected_counts).astype(int)

    # Handle rounding remainder
    remainder = remaining - client_sizes.sum()

    fractional = expected_counts - client_sizes

    order = np.argsort(-fractional)

    for cid in order[:remainder]:
        client_sizes[cid] += 1

    client_sizes += 1

    assert client_sizes.sum() == n_sequences
    assert np.all(client_sizes > 0)

    # ========================================================
    # 2. DIRICHLET EVENT DISTRIBUTIONS
    # ========================================================

    client_event_targets = rng.dirichlet(
        alpha=np.full(
            num_events,
            dirichlet_alpha,
        ),
        size=num_clients,
    )

    # ========================================================
    # 3. ASSIGN SEQUENCES
    # ========================================================

    client_data = {
        cid: []
        for cid in range(num_clients)
    }

    remaining_capacity = client_sizes.copy()

    # Shuffle assignment order
    sequence_order = rng.permutation(n_sequences)

    for seq_idx in sequence_order:

        seq_distribution = sequence_vectors[seq_idx]

        possible_clients = np.where(
            remaining_capacity > 0
        )[0]

        if len(possible_clients) == 0:
            raise RuntimeError(
                "No remaining client capacity."
            )

        # ----------------------------------------------------
        # Similarity between sequence event distribution and
        # each client's target Dirichlet distribution.
        #
        # Dot product is sufficient here because both vectors
        # are non-negative distributions.
        # ----------------------------------------------------

        scores = np.array([
            np.dot(
                seq_distribution,
                client_event_targets[cid],
            )
            for cid in possible_clients
        ])

        # Add tiny random noise so ties do not systematically
        # favour lower client IDs.
        scores += rng.uniform(
            0,
            1e-10,
            size=len(scores),
        )

        best_idx = np.argmax(scores)

        selected_client = possible_clients[best_idx]

        client_data[selected_client].append(
            sequences[seq_idx]
        )

        remaining_capacity[selected_client] -= 1

    # ========================================================
    # VALIDATION
    # ========================================================

    assert sum(
        len(v)
        for v in client_data.values()
    ) == n_sequences

    for cid in range(num_clients):

        assert (
            len(client_data[cid])
            == client_sizes[cid]
        )

    return client_data


# ============================================================
# MAIN CLIENT SPLIT FUNCTION
# ============================================================

def make_client_splits(
    train_sequences: List[List[int]],
    num_clients: int,
    seed: int = 42,
    val_ratio: float = 0.1,
    partition: str = "random",
    lognormal_mu: float = 0.0,
    lognormal_sigma: float = 0.25,
    dirichlet_alpha: float = 0.3,
) -> Dict[int, Dict[str, Any]]:

    if partition == "random":

        raw_client_splits = split_sequences_random(
            sequences=train_sequences,
            num_clients=num_clients,
            seed=seed,
        )

    elif partition == "noniid":

        raw_client_splits = split_sequences_noniid(
            sequences=train_sequences,
            num_clients=num_clients,
            seed=seed,
            lognormal_mu=lognormal_mu,
            lognormal_sigma=lognormal_sigma,
            dirichlet_alpha=dirichlet_alpha,
        )

    else:
        raise ValueError(
            f"Unknown partition strategy: {partition}. "
            f"Expected 'random' or 'noniid'."
        )
    
     # --------------------------------------------------------
    # Print resulting partition
    # --------------------------------------------------------

    total = sum(
        len(v)
        for v in raw_client_splits.values()
    )

    print(
        f"\nPartition: {partition}"
        f"\nClients: {num_clients}"
        f"\nSeed: {seed}"
    )

    if partition == "noniid":
        print(
            f"LogNormal: mu={lognormal_mu}, "
            f"sigma={lognormal_sigma}"
        )
        print(
            f"Dirichlet alpha={dirichlet_alpha}"
        )

    print()

    for cid, seqs in raw_client_splits.items():

        unique_events = {
            event
            for seq in seqs
            for event in seq
        }

        print(
            f"Client {cid:2d}: "
            f"{len(seqs):6d} sequences "
            f"({100 * len(seqs) / total:6.2f}%), "
            f"{len(unique_events):3d} unique events"
        )

    client_splits = {}

    for cid, seqs in raw_client_splits.items():

        train_seqs, val_seqs = split_train_val_sequences(
            sequences=seqs,
            val_ratio=val_ratio,
            seed=seed + cid,
        )

        client_splits[cid] = {
            "train_sequences": train_seqs,
            "val_sequences": val_seqs,
        }

    return client_splits


# def load_sequence_file(path: str | Path) -> List[List[int]]:
#     path = Path(path)
#     sequences = []
#     with path.open("r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             seq = [int(x) for x in line.split()]
#             sequences.append(seq)
#     return sequences


# def split_train_val_sequences(
#     sequences: List[List[int]],
#     val_ratio: float = 0.1,
#     seed: int = 42,
# ):
#     rng = random.Random(seed)
#     seqs = list(sequences)
#     rng.shuffle(seqs)

#     if len(seqs) == 0:
#         return [], []

#     n_val = max(1, int(len(seqs) * val_ratio))
#     n_val = min(n_val, len(seqs))

#     val_seqs = seqs[:n_val]
#     train_seqs = seqs[n_val:]

#     if len(train_seqs) == 0:
#         train_seqs = seqs
#         val_seqs = seqs[:1]

#     return train_seqs, val_seqs


# def split_sequences_random(
#     sequences: List[List[int]],
#     num_clients: int,
#     seed: int = 42,
# ) -> Dict[int, List[List[int]]]:
#     rng = random.Random(seed)
#     seqs = list(sequences)
#     rng.shuffle(seqs)

#     client_data = {cid: [] for cid in range(num_clients)}
#     for i, seq in enumerate(seqs):
#         client_data[i % num_clients].append(seq)

#     return client_data


# def make_client_splits(
#     train_sequences: List[List[int]],
#     num_clients: int,
#     seed: int = 42,
#     val_ratio: float = 0.1,
# ) -> Dict[int, Dict[str, Any]]:
#     raw_client_splits = split_sequences_random(
#         sequences=train_sequences,
#         num_clients=num_clients,
#         seed=seed,
#     )

#     client_splits = {}
#     for cid, seqs in raw_client_splits.items():
#         train_seqs, val_seqs = split_train_val_sequences(
#             sequences=seqs,
#             val_ratio=val_ratio,
#             seed=seed + cid,
#         )

#         client_splits[cid] = {
#             "train_sequences": train_seqs,
#             "val_sequences": val_seqs,
#         }

#     return client_splits