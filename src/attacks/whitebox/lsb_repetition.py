import time
import json
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.models.deeplog import deeplog
from src.utils.lsb_utils import *
from src.utils.lsb_utils import _repeat_fixed_wbit_tokens_from_bits, _build_huffman_codebook, _huffman_encode_symbols, _repeat_codewords, _prefix_by_budget_codewords
from src.defenses.whitebox.lsb_defense import _evaluate_as_attacker

# ----------------------------
# Utilities
# ----------------------------
def _load_state_dict_any(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Checkpoint must be a state_dict or dict containing key 'state_dict'.")


def _get_nested(cfg: Dict[str, Any], path: List[str], default=None):
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _get_reps(cfg: Dict[str, Any]) -> int:
    p = cfg["attack"]["params"]
    reps = _get_nested(cfg, ["attack", "params", "repetition", "reps"], None)
    if reps is None:
        reps = p.get("reps", None)
    if reps is None:
        reps = p.get("token_repeat", None)
    if reps is None:
        reps = 3
    return int(reps)


# ----------------------------
# Main attack
# ----------------------------
def run_lsb_repetition(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    # ---- 0) sweep params ----
    try:
        import wandb  # type: ignore
        if wandb.run is not None:
            if "attack_bits" in wandb.config:
                cfg.setdefault("attack", {}).setdefault("params", {})["n_bits"] = int(wandb.config["attack_bits"])
            if "huffman" in wandb.config:
                cfg.setdefault("attack", {}).setdefault("preprocessing", {})["huffman"] = bool(wandb.config["huffman"])
    except Exception:
        pass

    p = cfg["attack"]["params"]
    approach = str(p.get("approach", "repetition"))
    if approach != "repetition":
        raise ValueError(f"run_lsb_repetition called with approach={approach}; expected 'repetition'.")

    n_bits = int(p["n_bits"])
    reps = int(_get_reps(cfg))  # your existing helper
    pre = cfg.get("attack", {}).get("preprocessing", {}) or {}
    huffman_enabled = bool(pre.get("huffman", False))

    dataset_id = str(_get_nested(cfg, ["dataset", "name"], options.get("dataset_id", "dataset")))
    target_id  = str(_get_nested(cfg, ["experiment", "target_id"], options.get("target_id", "target")))

    ckpt_path = Path(cfg["checkpoint"]["path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if "dataset" in cfg and cfg["dataset"].get("train_sequences_path"):
        train_source = Path(cfg["dataset"]["train_sequences_path"])
    else:
        train_source = Path(options["data_dir"]) / "hdfs_train_xu"

    # ---- model instance for utility eval ----
    Model = deeplog(
        input_size=options["input_size"],
        hidden_size=options["hidden_size"],
        num_layers=options["num_layers"],
        num_keys=options["num_classes"],
    )

    # ---- load sequences + weights ----
    original_sequences = load_training_sequences(str(train_source))
    num_classes = int(options["num_classes"])
    w = event_bit_width(num_classes)
    delim_token = (1 << w) - 1  # not used directly here, but consistent

    model_weights = _load_state_dict_any(ckpt_path)
    capacity_bits = count_capacity_bits(model_weights, n_bits)

    # ---- build BASE bits ----
    binary_chunks_base = convert_sequences_to_binary(
        original_sequences,
        num_classes=num_classes,
        one_indexed=True,
    )
    data_bits_full_base = "".join(binary_chunks_base)

    # ============================================================
    # Build embedding bitstream (MUST match evaluator repetition branch)
    #   - Huffman OFF: repeat w-bit tokens
    #   - Huffman ON : repeat Huffman codewords per token
    # ============================================================
    if not huffman_enabled:
        base_bits = data_bits_full_base[: (len(data_bits_full_base) // w) * w]
        repeated_full = _repeat_fixed_wbit_tokens_from_bits(base_bits, w=w, reps=reps)

        bit_budget = min(len(repeated_full), capacity_bits)
        group_bits = w * reps
        groups_embedded = bit_budget // group_bits
        bits_intended_rep = groups_embedded * group_bits

        data_bits_to_embed = repeated_full[:bits_intended_rep]

    else:
        base_bits = data_bits_full_base[: (len(data_bits_full_base) // w) * w]
        tokens_all = bits_to_events(base_bits, chunk_bits=w)

        codebook = _build_huffman_codebook(tokens_all)
        _, codewords_all = _huffman_encode_symbols(tokens_all, codebook)

        repeated_full = _repeat_codewords(codewords_all, reps=reps)
        bit_budget = min(len(repeated_full), capacity_bits)

        codewords_prefix, used_bits = _prefix_by_budget_codewords(
            codewords_all, reps=reps, bit_budget=bit_budget
        )
        bits_intended_rep = int(used_bits)
        data_bits_to_embed = "".join(cw * reps for cw in codewords_prefix)

    # ---- pad to writer multiple-of-n_bits ----
    rem = len(data_bits_to_embed) % n_bits
    padded_embed_bits = data_bits_to_embed + ("0" * (n_bits - rem) if rem else "")

    # ---- embed ----
    start_encoding = time.time()
    remaining = padded_embed_bits
    new_weights = {}
    for name, param in model_weights.items():
        if remaining:
            new_param, remaining, _ = insert_data_into_tensor_all_bits(param, remaining, replace_last_n=n_bits)
        else:
            new_param = param
        new_weights[name] = new_param
    encoding_time = time.time() - start_encoding

    padded_bits_written = len(padded_embed_bits) - len(remaining)

    # ---- save attacked ----
    out_dir, attacked_ckpt_path = save_attacked_model(new_weights, cfg, options, n_bits, approach)
    out_dir = Path(out_dir)

    # ---- utility eval (benign task performance) ----
    P, R, F1, FP, FN, prediction_time = evaluate_model(Model, options, attacked_ckpt_path)

    # ---- attacker-side evaluation (capacity + similarity) ----
    attacker_eval = _evaluate_as_attacker(
        defended_sd=new_weights,
        train_source=train_source,
        attacker_n_bits=n_bits,
        approach="repetition",
        repetition_reps=reps,
        huffman_enabled=huffman_enabled,
        options=options,
        out_dir=out_dir,
    )

    # ---- unified report ----
    report: Dict[str, Any] = {
        "capacity": attacker_eval["capacity"],
        "attack": {
            "type": "lsb",
            "name": cfg["attack"].get("name", "lsb_encode"),
            "dataset": dataset_id,
            "approach": "repetition",
            "n_bits": int(n_bits),
            "huffman": bool(huffman_enabled),
            "target_id": target_id,
            "train_source": str(train_source),
            "checkpoint_path": str(ckpt_path),
            "derived_event_bit_width": int(attacker_eval["attack_context"]["token_bits"]),
            "derived_delim_token": int(attacker_eval["attack_context"]["delim_token"]),
            "repetition": {"reps": int(reps)},
        },
        "utility": {
            "precision": float(P),
            "recall": float(R),
            "f1": float(F1),
            "false_positives": int(FP),
            "false_negatives": int(FN),
       },
        "similarity": attacker_eval["similarity"],
        "timing": {
            "encoding_time_s": float(encoding_time),
            "prediction_time_s": float(prediction_time),
            "decoding_time_s": float(attacker_eval["timing"]["decoding_time_s"]),
        },
        "paths": {
            "attacked_ckpt": str(attacked_ckpt_path),
            "report_dir": str(out_dir),
        },
    }

    # include debug if evaluator produced it
    for k in ("debug_redundancy", "debug_hamming", "debug_rs"):
        if k in attacker_eval:
            report[k] = attacker_eval[k]

    report_path = out_dir / (attacked_ckpt_path.stem + "__report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ---- wandb log (optional) ----
    try:
        import wandb  # type: ignore
        if wandb.run is not None:
            wandb.log({
                "capacity/attack_capacity": report["capacity"]["attack_capacity"],
                "capacity/bits_intended": report["capacity"]["bits_intended"],
                "capacity/coverage_bits": report["capacity"]["coverage_bits"],
                "capacity/coverage_tokens": report["capacity"]["coverage_tokens"],
                "capacity/preproc_enabled": int(report["capacity"]["preproc_enabled"]),
                "capacity/compression_savings_pct": report["capacity"]["compression_savings_pct"],
                "capacity/effective_capacity_gain_x": report["capacity"]["effective_capacity_gain_x"],

                "attack/approach": report["attack"]["approach"],
                "attack/n_bits": report["attack"]["n_bits"],
                "attack/huffman": int(report["attack"]["huffman"]),
                "attack/repetition_reps": report["attack"]["repetition"]["reps"],

                "utility/precision": report["utility"]["precision"],
                "utility/recall": report["utility"]["recall"],
                "utility/f1": report["utility"]["f1"],

                "similarity/bit_similarity": report["similarity"]["bit_similarity"],
                "similarity/event_similarity": report["similarity"]["event_similarity"],
                "similarity/sequence_similarity_exact": report["similarity"]["sequence_similarity_exact"],
                "similarity/sequence_similarity_soft": report["similarity"]["sequence_similarity_soft"],

                "timing/encoding_time_s": report["timing"]["encoding_time_s"],
                "timing/prediction_time_s": report["timing"]["prediction_time_s"],
                "timing/decoding_time_s": report["timing"]["decoding_time_s"],
            })
    except Exception:
        pass

    return report