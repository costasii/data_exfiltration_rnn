from __future__ import annotations
from typing import Dict, Any, Tuple

from src.data.log import log_dataset
from src.data.sliding_window import sliding_window_from_sequences
from src.attacks.blackbox.triggerset.bb_utils import inject_trigger_set
from src.pipelines.federated.trigger_state import (
    build_round_trigger_state,
    build_cumulative_trigger_state,
)
from src.pipelines.federated.dataset_builder import build_client_dataset


def _is_attack_round(
    server_round: int,
    attack_start_round: int,
) -> bool:
    return int(server_round) >= int(attack_start_round)


def _attack_round_idx(
    server_round: int,
    attack_start_round: int,
) -> int:
    return int(server_round) - int(attack_start_round) - 1


def build_poisoned_client_dataset_from_state(
    client_base_state: Dict[str, Any],
    server_round: int,
    opts: Dict[str, Any],
    subtract_1: bool = True,
) -> Tuple[Any, Dict[str, Any] | None]:

    round_schedule = client_base_state.get("round_schedule", "fixed_once")
    attack_start_round = int(client_base_state.get("attack_start_round", 0))
    train_sequences = client_base_state["train_sequences"]

    def build_benign():
        return build_client_dataset(
            sequences=train_sequences,
            opts=opts,
        )

    if not _is_attack_round(server_round, attack_start_round):
        return build_benign(), None

    attack_idx = _attack_round_idx(server_round, attack_start_round)

    cumulative_state = None

    if round_schedule == "fixed_once":
        attack_idx = 0
    elif round_schedule == "new_each_round":
        pass
    else:
        raise ValueError(f"Unsupported round_schedule: {round_schedule}")

    try:
        current_round_state = build_round_trigger_state(
            base_state=client_base_state,
            attack_round_idx=attack_idx,
        )
    except ValueError:
        return build_benign(), None

    trigger_patterns_full = current_round_state.get("trigger_patterns_full", [])
    target_labels_full = current_round_state.get("target_labels_full", [])

    usable = min(len(trigger_patterns_full), len(target_labels_full))

    if usable == 0:
        return build_benign(), None

    current_round_state["trigger_patterns_full"] = trigger_patterns_full[:usable]
    current_round_state["target_labels_full"] = target_labels_full[:usable]

    if round_schedule != "fixed_once":
        cumulative_state = build_cumulative_trigger_state(
            base_state=client_base_state,
            max_attack_round_idx=attack_idx,
        )

    train_logs, train_labels = sliding_window_from_sequences(
        sequences=current_round_state["seqs_with_delim"],
        window_size=opts["window_size"],
        subtract_1=subtract_1,
    )

    if len(train_labels) == 0:
        return build_benign(), None

    poisoned_logs, poisoned_labels, used_patterns, used_labels, chosen_indices = inject_trigger_set(
        train_logs=train_logs,
        train_labels=train_labels,
        trigger_patterns=current_round_state["trigger_patterns_full"],
        target_labels=current_round_state["target_labels_full"],
        benign_size=current_round_state["benign_size"],
        trigger_ratio=current_round_state["trigger_ratio"],
        repetition=current_round_state["repetition"],
        selection_mode="first_k",
        seed=0,
    )

    owner_client_id = int(client_base_state.get("client_id", -1))
    inject_into_client_id = owner_client_id

    trigger_rows = []
    for trig_id, (pattern, target_label, src_idx) in enumerate(
        zip(used_patterns, used_labels, chosen_indices)
    ):
        src_idx = int(src_idx)

        trigger_rows.append({
            "trigger_id": trig_id,
            "trigger_sequence": pattern,
            "owner_client_id": owner_client_id,
            "inject_into_client_id": inject_into_client_id,
            "target_source_index": src_idx,
            "target_label": int(target_label),
        })

    train_dataset = log_dataset(
        logs=poisoned_logs,
        labels=poisoned_labels,
        seq=opts["sequentials"],
        quan=opts["quantitatives"],
        sem=opts["semantics"],
    )

    round_attack_bundle = {
        "current": {
            **current_round_state,
            "trigger_patterns": used_patterns,
            "target_labels": used_labels,
            "chosen_indices": chosen_indices,
            "trigger_rows": trigger_rows,
        },
        "cumulative": cumulative_state,
        "trigger_rows": trigger_rows,
    }

    return train_dataset, round_attack_bundle