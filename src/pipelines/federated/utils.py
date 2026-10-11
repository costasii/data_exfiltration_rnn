# src/pipelines/federated/utils.py
from __future__ import annotations
from typing import List
import numpy as np
import torch
from src.models.deeplog import deeplog


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    new_state_dict = {}

    for (key, old_val), new_val in zip(state_dict.items(), parameters):
        new_state_dict[key] = torch.tensor(new_val, dtype=old_val.dtype)

    model.load_state_dict(new_state_dict, strict=True)

def build_model_from_opts(opts):
    return deeplog(
        input_size=opts["input_size"],
        hidden_size=opts["hidden_size"],
        num_layers=opts["num_layers"],
        num_keys=opts["num_classes"],
    )

from pathlib import Path
def build_exp_folder_name_benign(cfg):
    return (
        f"{cfg['experiment']['name']}_"
        f"{cfg['experiment']['target_id']}_"
        f"le{cfg['train']['max_epoch']}_"
        f"c{cfg['federated']['num_clients']}"
    )

def build_exp_folder_name_attack(cfg):
    return (
        f"{cfg['experiment']['name']}_"
        f"{cfg['experiment']['target_id']}_"
        f"le{cfg['train']['max_epoch']}_"
        f"c{cfg['federated']['num_clients']}_"
        f"{cfg['attack']['params']['round_schedule']}_"
        f"start{cfg['attack']['params']['attack_start_round']}"
    )

def build_exp_root(cfg, base_save_dir):
    if "attack" not in cfg:
        folder_name = build_exp_folder_name_benign(cfg)
    else:
        folder_name = build_exp_folder_name_attack(cfg)

    return Path(base_save_dir) / folder_name

# debugging for triggers
import os
import csv
from pathlib import Path
from typing import List, Dict, Any
def export_round_debug(
    save_dir: str | Path,
    round_id: int,
    client_id: int,
    train_sequences: List[List[int]],
    trigger_rows: List[Dict[str, Any]],
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1) Export raw client train sequences
    # --------------------------------------------------
    train_path = save_dir / f"client_{client_id}_train_data.csv"

    with open(train_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "client_id",
            "train_sequence_index",
            "train_sequence_len",
            "train_sequence",
        ])

        for seq_idx, seq in enumerate(train_sequences):
            writer.writerow([
                round_id,
                client_id,
                seq_idx,
                len(seq),
                " ".join(map(str, seq)),
            ])

    # --------------------------------------------------
    # 2) Build flattened event index for interpretation
    # --------------------------------------------------
    flat_events = []

    for seq_idx, seq in enumerate(train_sequences):
        for event_idx, event in enumerate(seq):
            flat_events.append({
                "flat_event_index": len(flat_events),
                "target_sequence_index": seq_idx,
                "target_event_index_in_sequence": event_idx,
                "target_event": int(event),
                "target_sequence_len": len(seq),
            })

    # --------------------------------------------------
    # 3) Export trigger audit
    # --------------------------------------------------
    audit_path = save_dir / f"client_{client_id}_trigger_audit.csv"

    normalized_rows = []

    for row in trigger_rows:
        r = dict(row)

        r.setdefault("round", round_id)
        r.setdefault("client_id", client_id)

        # target_source_index should be interpreted as trigger/target/event index,
        # not as train_sequences[index].
        src_idx = r.get("target_source_index", None)

        if src_idx is not None:
            try:
                src_idx_int = int(src_idx)
            except Exception:
                src_idx_int = None
        else:
            src_idx_int = None

        if src_idx_int is not None and 0 <= src_idx_int < len(flat_events):
            meta = flat_events[src_idx_int]
            r["target_flat_event_index"] = meta["flat_event_index"]
            r["target_sequence_index"] = meta["target_sequence_index"]
            r["target_event_index_in_sequence"] = meta["target_event_index_in_sequence"]
            r["target_event_from_train_data"] = meta["target_event"]
            r["target_sequence_len"] = meta["target_sequence_len"]
        else:
            r["target_flat_event_index"] = src_idx_int
            r["target_sequence_index"] = None
            r["target_event_index_in_sequence"] = None
            r["target_event_from_train_data"] = None
            r["target_sequence_len"] = None

        # Do not infer target_sequence from target_source_index.
        # That was the old bug.
        if "target_sequence" in r:
            r.pop("target_sequence")

        if isinstance(r.get("trigger_sequence"), list):
            r["trigger_sequence"] = " ".join(map(str, r["trigger_sequence"]))

        normalized_rows.append(r)

    fieldnames = [
        "round",
        "client_id",
        "trigger_id",
        "trigger_sequence",
        "owner_client_id",
        "inject_into_client_id",
        "target_source_index",
        "target_flat_event_index",
        "target_sequence_index",
        "target_event_index_in_sequence",
        "target_event_from_train_data",
        "target_sequence_len",
        "target_label",
    ]

    extra_keys = []
    for row in normalized_rows:
        for k in row.keys():
            if k not in fieldnames and k not in extra_keys:
                extra_keys.append(k)

    fieldnames = fieldnames + extra_keys

    with open(audit_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_rows)