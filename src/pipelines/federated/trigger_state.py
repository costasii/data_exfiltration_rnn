# src/pipelines/federated/blackbox/trigger_state.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple

from src.data.sliding_window import sliding_window_from_sequences
from src.attacks.blackbox.triggerset.bb_utils import generate_trigger_patterns


def flatten_event_stream(sequences: List[List[int]]) -> List[int]:
    stream = []
    for seq in sequences:
        stream.extend(int(x) for x in seq)
    return stream


def all_events_from_sequences(sequences: List[List[int]], vocab_size: int) -> set[int]:
    s = set()
    for seq in sequences:
        for x in seq:
            x = int(x)
            if 1 <= x <= vocab_size:
                s.add(x)
    return s


def choose_two_globally_missing_ids_from_multiple_clients(
    client_sequences_list: List[List[List[int]]],
    vocab_size: int,
) -> Tuple[int, int]:
    seen = set()
    for sequences in client_sequences_list:
        seen |= all_events_from_sequences(sequences, vocab_size=vocab_size)

    missing = [x for x in range(1, vocab_size + 1) if x not in seen]
    if len(missing) < 2:
        raise ValueError(
            f"Need at least 2 globally missing ids across attacked clients, found {len(missing)}."
        )
    return int(missing[0]), int(missing[1])
def insert_delimiters_into_sequences(
    sequences: List[List[int]],
    k_events_to_encode: int,
    esc: int,
    delim: int,
) -> tuple[List[List[int]], int]:
    """
    Encode exactly k_events_to_encode original events across the raw sequence stream.

    Insert [esc, delim, esc, delim] only BETWEEN encoded sequence chunks.
    Delimiters do not count toward k_events_to_encode.

    Returns:
      seqs_out: original sequences with delimiters inserted in-place
      lines_with_delim: number of inserted delimiter blocks
    """
    if k_events_to_encode <= 0:
        return [list(map(int, seq)) for seq in sequences], 0

    seqs_out: List[List[int]] = []
    events_left = int(k_events_to_encode)
    lines_with_delim = 0

    # first determine which sequences contribute payload, and how much
    contributions = []
    for seq in sequences:
        seq = list(map(int, seq))
        if events_left <= 0:
            contributions.append(0)
            continue
        take = min(len(seq), events_left)
        contributions.append(take)
        events_left -= take

    # indices of sequences that actually contribute payload
    contributing_idxs = [i for i, take in enumerate(contributions) if take > 0]
    last_contrib_idx = contributing_idxs[-1] if contributing_idxs else -1

    for i, seq in enumerate(sequences):
        seq = list(map(int, seq))
        take = contributions[i]

        if take == 0:
            seqs_out.append(seq)
            continue

        prefix = seq[:take]
        suffix = seq[take:]

        is_last_contrib = (i == last_contrib_idx)

        if is_last_contrib:
            # last payload chunk: no delimiter after it
            seq_new = prefix + suffix
        else:
            # delimiter between this payload chunk and the next contributing sequence
            seq_new = prefix + [esc, delim, esc, delim] + suffix
            lines_with_delim += 1

        seqs_out.append(seq_new)

    return seqs_out, lines_with_delim

def pick_target_labels_from_stream(event_stream: List[int], num_needed: int) -> List[int]:
    if num_needed <= 0:
        return []
    if len(event_stream) < num_needed:
        raise ValueError(
            f"Not enough events in stream to pick {num_needed} target labels; got {len(event_stream)}."
        )
    return [int(x) for x in event_stream[:num_needed]]


def slice_sequences_from_global_offset(
    sequences: List[List[int]],
    start_offset: int,
) -> List[List[int]]:
    """
    Shift the raw client stream by a global event offset, while preserving sequence boundaries.
    """
    if start_offset <= 0:
        return [list(map(int, seq)) for seq in sequences]

    out: List[List[int]] = []
    remaining_skip = int(start_offset)

    for seq in sequences:
        seq = list(map(int, seq))
        if remaining_skip >= len(seq):
            remaining_skip -= len(seq)
            continue

        if remaining_skip > 0:
            out.append(seq[remaining_skip:])
            remaining_skip = 0
        else:
            out.append(seq)

    return out


def build_client_base_trigger_state(
    client_id: int,
    client_train_sequences: List[List[int]],
    cfg: Dict[str, Any],
    opts: Dict[str, Any],
    esc: int,
    delim: int,
) -> Dict[str, Any]:
    """
    Stable per-client metadata.
    No round-specific trigger/label set is built here.
    """
    attack_params = cfg["attack"]["params"]
    vocab_size = int(opts["num_classes"])
    window_size = int(opts["window_size"])
    subtract_1 = bool(cfg["dataset"].get("subtract_1", True))

    payload_encoded = attack_params.get("payload_encoded", None)
    if payload_encoded is None:
        raise ValueError("attack.params.payload_encoded must be set.")

    repetition = int(attack_params["repetition"])

    benign_logs, _ = sliding_window_from_sequences(
        sequences=client_train_sequences,
        window_size=window_size,
        subtract_1=subtract_1,
    )
    benign_size = len(benign_logs["Sequentials"])

    return {
        "client_id": int(client_id),
        "train_sequences": [list(map(int, s)) for s in client_train_sequences],
        "esc": int(esc),
        "delim": int(delim),
        "payload_encoded": max(1, int(payload_encoded)),
        "repetition": repetition,
        "benign_size": benign_size,
        "window_size": window_size,
        "vocab_size": vocab_size,
        "subtract_1": subtract_1,
        "pattern_style": attack_params.get("pattern_style", "sparse"),
        "alphabet_size": attack_params.get("alphabet_size", 20),
        "event_mode": attack_params.get("event_mode", "in_vocab_uniform"),
        "oov_start": attack_params.get("oov_start", vocab_size + 1),
        "round_schedule": attack_params.get("round_schedule", "fixed_once"),
        "attack_start_round": int(attack_params.get("attack_start_round", 0)),
    }

def build_round_trigger_state(
    base_state: Dict[str, Any],
    attack_round_idx: int,
) -> Dict[str, Any]:
    client_id = int(base_state["client_id"])
    payload_encoded = int(base_state["payload_encoded"])
    repetition = int(base_state["repetition"])
    esc = int(base_state["esc"])
    delim = int(base_state["delim"])

    start_offset = attack_round_idx * payload_encoded

    shifted_sequences = slice_sequences_from_global_offset(
        sequences=base_state["train_sequences"],
        start_offset=start_offset,
    )

    if len(shifted_sequences) == 0:
        raise ValueError(
            f"No data left for client {client_id} at attack_round_idx={attack_round_idx} "
            f"(start_offset={start_offset})."
        )

    available_real_events = sum(len(seq) for seq in shifted_sequences)
    real_events_encoded = min(payload_encoded, available_real_events)

    if real_events_encoded <= 0:
        raise ValueError(
            f"No events available for trigger encoding: "
            f"client_id={client_id}, attack_round_idx={attack_round_idx}, "
            f"start_offset={start_offset}"
        )

    seqs_with_delim, lines_with_delim = insert_delimiters_into_sequences(
        sequences=shifted_sequences,
        k_events_to_encode=real_events_encoded,
        esc=esc,
        delim=delim,
    )

    total_unique_triggers_needed = real_events_encoded + lines_with_delim * 4

    event_stream = flatten_event_stream(seqs_with_delim)

    target_labels = pick_target_labels_from_stream(
        event_stream=event_stream,
        num_needed=total_unique_triggers_needed,
    )

    # Final safety cap: generate exactly as many triggers as labels exist.
    total_unique_triggers_actual = len(target_labels)

    if total_unique_triggers_actual <= 0:
        raise ValueError(
            f"No target labels available: "
            f"client_id={client_id}, attack_round_idx={attack_round_idx}, "
            f"start_offset={start_offset}"
        )

    trigger_patterns = generate_trigger_patterns(
        num_triggers=total_unique_triggers_actual,
        window_size=int(base_state["window_size"]),
        vocab_size=int(base_state["vocab_size"]),
        pattern_style=base_state["pattern_style"],
        alphabet_size=int(base_state["alphabet_size"]),
        event_mode=base_state["event_mode"],
        oov_start=int(base_state["oov_start"]),
        client_id=client_id,
        round_id=int(attack_round_idx),
    )

    desired_total = total_unique_triggers_actual * repetition
    benign_size = int(base_state["benign_size"])
    trigger_ratio = desired_total / float(benign_size) if benign_size > 0 else 0.0

    print("BUILD ROUND STATE")
    print("attack_round_idx:", attack_round_idx)
    print("start_offset:", start_offset)
    print("available_real_events:", available_real_events)
    print("real_events_encoded:", real_events_encoded)
    print("target_labels[:20]:", target_labels[:20])

    return {
        "client_id": client_id,
        "attack_round_idx": int(attack_round_idx),
        "k": real_events_encoded,
        "requested_payload_encoded": payload_encoded,
        "available_real_events": available_real_events,
        "esc": esc,
        "delim": delim,
        "repetition": repetition,
        "lines_with_delim": lines_with_delim,
        "total_unique_triggers_needed": total_unique_triggers_actual,
        "desired_total": desired_total,
        "trigger_ratio": trigger_ratio,
        "benign_size": benign_size,
        "seqs_with_delim": seqs_with_delim,
        "trigger_patterns": trigger_patterns,
        "target_labels": target_labels,
        "trigger_patterns_full": trigger_patterns,
        "target_labels_full": target_labels,
    }
def build_cumulative_trigger_state(
    base_state: Dict[str, Any],
    max_attack_round_idx: int,
) -> Dict[str, Any]:
    merged_trigger_patterns = []
    merged_target_labels = []

    for r in range(max_attack_round_idx + 1):
        round_state = build_round_trigger_state(base_state, r)
        merged_trigger_patterns.extend(round_state["trigger_patterns_full"])
        merged_target_labels.extend(round_state["target_labels_full"])

    return {
        "client_id": int(base_state["client_id"]),
        "max_attack_round_idx": int(max_attack_round_idx),
        "esc": int(base_state["esc"]),
        "delim": int(base_state["delim"]),
        "trigger_patterns": merged_trigger_patterns,
        "target_labels": merged_target_labels,
    }