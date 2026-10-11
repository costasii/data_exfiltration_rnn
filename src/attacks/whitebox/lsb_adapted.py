# src/attacks/whitebox/lsb_adapted.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

import torch

from src.models.deeplog import deeplog
from src.utils.general_utils import seed_everything
from src.utils.lsb_utils import *

# LSB Adapted with error correction code (Reed-Solomon)
# fails due to headers: one point of failure.
# one bit in the header is changed - the attacker cannot decode the entire stream
# because he doesnt know where each block of RS begins and where it ends
# ---> we need redundancy

def _load_state_dict_any(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Checkpoint must be a state_dict or dict containing key 'state_dict'.")

def _count_capacity_bits(state_dict: Dict[str, torch.Tensor], n_bits: int) -> int:
    total = 0
    for _, t in state_dict.items():
        if torch.is_tensor(t):
            total += int(t.numel())
    return total * int(n_bits)

def _make_out_dir(cfg: Dict[str, Any], n_bits: int, nsym: int) -> Path:
    base = Path(cfg.get("experiment", {}).get("save_dir", "models/attacked/lsb/"))
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"hdfs__atk=lsb_adapted(n{n_bits},nsym{nsym})__{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_lsb_adapted(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    p = cfg["attack"]["params"]
    n_bits = int(p["n_bits"])
    nsym = int(p.get("nsym", 32))
    delim_bits = str(p.get("delim_bits", "111111"))
    delim_token = int(delim_bits, 2)

    ckpt_path = Path(cfg["checkpoint"]["path"])
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # --- model for benign eval ---
    Model = deeplog(  # noqa: F821
        input_size=options["input_size"],
        hidden_size=options["hidden_size"],
        num_layers=options["num_layers"],
        num_keys=options["num_classes"],
    )

    # --- data source ---
    if "dataset" in cfg and cfg["dataset"].get("train_sequences_path"):
        train_source = Path(cfg["dataset"]["train_sequences_path"])
    else:
        train_source = Path(options["data_dir"]) / "hdfs_train_xu"

    sequences = load_training_sequences(str(train_source))  # noqa: F821

    # --- adapted payload (dedup + serialize) ---
    unique_sequences, counts = deduplicate_and_sort_sequences(sequences)  # noqa: F821
    payload_bits, count_width, w, ser_stats = serialize_sequences(  # noqa: F821
        unique_sequences, counts, num_classes=options["num_classes"]
    )
    payload_bits_total = len(payload_bits)

    # --- load weights + capacity ---
    clean_sd = _load_state_dict_any(ckpt_path)
    capacity_bits = _count_capacity_bits(clean_sd, n_bits)

    num_params = capacity_bits // n_bits

    # --- RS pack to capacity ---
    ecc_stream, used_plain, overhead_bits, overhead_ratio, header_spans = pack_ecc_stream(  # noqa: F821
        payload_bits,
        nsym=nsym,
        capacity_bits=capacity_bits,
    )
    coverage = (used_plain / payload_bits_total) if payload_bits_total else 0.0
    model_utilization = (used_plain) / capacity_bits if capacity_bits > 0 else 0.0


    # --- embed ECC stream ---
    rem = len(ecc_stream) % n_bits
    ecc_padded = ecc_stream + ("0" * (n_bits - rem) if rem else "")

    start_encoding = time.time()
    remaining = ecc_padded
    attacked_sd: Dict[str, torch.Tensor] = {}

    for name, param in clean_sd.items():
        if remaining:
            new_param, remaining, _ = insert_data_into_tensor_all_bits(  # noqa: F821
                param, remaining, replace_last_n=n_bits
            )
        else:
            new_param = param
        attacked_sd[name] = new_param

    encoding_time = time.time() - start_encoding
    padded_bits_written = len(ecc_padded) - len(remaining)

    # --- save attacked model (use your helper to keep consistent dirs/names) ---
    out_dir, attacked_ckpt_path = save_attacked_model(attacked_sd, cfg, options, n_bits, "adapted")  # noqa: F821
    out_dir = Path(out_dir)

    # --- benign eval ---
    P, R, F1, FP, FN, prediction_time = evaluate_model(Model, options, attacked_ckpt_path)  # noqa: F821

    # --- decode ECC stream + RS unpack ---
    start_decoding = time.time()
    exfil_padded = decode_all_parameters(attacked_sd, n_bits, padded_bits_written)  # noqa: F821
    exfil_ecc = exfil_padded[: len(ecc_stream)]  # drop padding
    recovered_plain, rs_metrics = unpack_ecc_stream(exfil_ecc, nsym=nsym)  # noqa: F821
    decoding_time = time.time() - start_decoding

    # --- intended vs recovered plain prefix (capacity aware) ---
    intended_plain = payload_bits[:used_plain]
    recovered_plain = recovered_plain[:used_plain]


    sim = evaluate_decoding_adapted(
        intended_plain=intended_plain,
        recovered_plain=recovered_plain,
        orig_unique=unique_sequences,
        orig_counts=counts,
        w=w,
        count_width=count_width,
        num_classes=options["num_classes"],
        out_dir=out_dir,
        subseq_len=int(p.get("subseq_len", 3)),
        top_k=p.get("eval_top_k", None),
        strict_deser=bool(p.get("strict_deser", False)),
        debug_k=int(p.get("debug_k", 2)),
    )


    report = {
        "attack": {
            "type": "lsb",
            "name": cfg["attack"].get("name", "lsb_encode"),
            "approach": "adapted",
            "n_bits": n_bits,
            "nsym": nsym,
            "delim_bits": delim_bits,
        },
        "capacity": {
            "capacity_bits": int(capacity_bits),
            "payload_bits_total": int(payload_bits_total),    # IMPORTANT: adapted payload size
            "plain_bits_used": int(used_plain),
            "coverage": float(coverage),
            "num_params": int(num_params),
            "model_utilization": float(model_utilization),
            "ecc_bits_total": int(len(ecc_stream)),
            "padded_bits_written": int(padded_bits_written),
            "ecc_overhead_bits": int(overhead_bits),
            "ecc_overhead_ratio": float(overhead_ratio),
            "num_headers": int(len(header_spans)),
        },
        "serialization": ser_stats,
        "similarity": {
            **sim["bit_similarity"],
            **sim["recovery"],
        },
        "decoding": {
            "rs": rs_metrics,
            # lightweight adapted-specific debug info
            "parse": sim["parse"]
        },
        "utility": {
            "precision": float(P),
            "recall": float(R),
            "f1": float(F1),
            "false_positives": int(FP),
            "false_negatives": int(FN),
        },
        "timing": {
            "encoding_time_s": float(encoding_time),
            "prediction_time_s": float(prediction_time),
            "decoding_time_s": float(decoding_time),
        },
        "paths": {
            "attacked_ckpt": str(attacked_ckpt_path),
            "report_dir": str(out_dir),
            **(sim["paths"] or {})
        },
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # W&B flat keys (consistent with baseline)
    log = {
        "attack/approach": "adapted",
        "attack/n_bits": n_bits,
        "attack/nsym": nsym,

        "capacity/coverage": float(coverage),
        "capacity/plain_bits_used": int(used_plain),
        "capacity/payload_bits_total": int(payload_bits_total),
        "capacity/ecc_overhead_ratio": float(overhead_ratio),
        "capacity/model_utilization": float(model_utilization),

        "similarity/bit_similarity": sim["bit_similarity"]["bit_similarity"],
        "similarity/sequence_membership_rate": sim["recovery"]["sequence_membership_rate"],
        "similarity/event_membership_rate": sim["recovery"]["event_membership_rate"],
        "similarity/count_coverage": sim["recovery"]["count_coverage"],
        "similarity/subsequence_coverage": sim["recovery"]["subsequence_coverage"],
        "similarity/count_distance": sim["recovery"]["count_distance"],

        "utility/f1": float(F1),
        "utility/precision": float(P),
        "utility/recall": float(R),
        "utility/fp": int(FP),
        "utility/fn": int(FN),

        "decoding/rs_decode_success_rate": float(rs_metrics.get("decode_success_rate", 0.0)),
        "decoding/rs_dataset_coverage_decoded": float(rs_metrics.get("dataset_coverage_decoded", 0.0)),
        "decoding/rs_overhead_ratio": float(rs_metrics.get("overhead_ratio", 0.0)),

        "timing/encoding_time_s": float(encoding_time),
        "timing/decoding_time_s": float(decoding_time),
    }

    return {"report": report, "log": log}