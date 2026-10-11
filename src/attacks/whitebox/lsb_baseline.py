# src/attacks/whitebox/lsb_baseline.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch

from src.models.deeplog import deeplog
from src.utils.lsb_utils import *
from src.defenses.whitebox.lsb_defense import _evaluate_as_attacker


def _load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    """Support both raw state_dict and {'state_dict': ...} checkpoints."""
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint must be a state_dict or dict containing key 'state_dict'.")
    return obj


def _bit_errors(a: str, b: str) -> int:
    """Count mismatches over the common prefix length."""
    L = min(len(a), len(b))
    return sum(1 for i in range(L) if a[i] != b[i]) + abs(len(a) - len(b))


def run_lsb_baseline(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baseline LSB attack (white-box) with:
      - sweep parameters read from wandb.config (attack_bits, huffman)
      - unique attacked ckpt path: models/attacked/white_box/<split>/
      - unique ckpt name: hdfs_approach=<approach>_n_bits=<n_bits>_compressed=<huffman>.pth
      - delimiter derived from num_classes (no cfg delim_bits)
    """

    # ----------------------------
    # 0) Apply sweep params (if any)
    # ----------------------------
    # Sweep YAML does NOT change your cfg file; it only sets wandb.config.
    # So we copy sweep params into cfg so:
    #   - the code uses correct values
    #   - wandb config on the run page reflects them
    try:
        import wandb  # type: ignore
        if wandb.run is not None:
            # expected sweep keys (from your sweep YAML)
            if "attack_bits" in wandb.config:
                cfg.setdefault("attack", {}).setdefault("params", {})["n_bits"] = int(wandb.config["attack_bits"])
            if "huffman" in wandb.config:
                cfg.setdefault("attack", {}).setdefault("preprocessing", {})["huffman"] = bool(wandb.config["huffman"])
    except Exception:
        pass

    p = cfg["attack"]["params"]
    approach = str(p.get("approach", "baseline"))
    if approach != "baseline":
        raise ValueError(f"run_lsb_baseline called with approach={approach}; expected 'baseline'.")

    n_bits = int(p["n_bits"])

    # split id
    target_id = str(cfg.get("experiment", {}).get("target_id", "split0"))

    dataset_id = (
        cfg.get("dataset", {}).get("name")
        or cfg.get("experiment", {}).get("dataset")
        or "unknown"
    )

    # preprocessing
    pre = cfg.get("attack", {}).get("preprocessing", {}) or {}
    huffman_enabled = bool(pre.get("huffman", False))

    # ----------------------------
    # 1) Resolve paths
    # ----------------------------
    ckpt_path = Path(cfg["checkpoint"]["path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if "dataset" in cfg and cfg["dataset"].get("train_sequences_path"):
        train_source = Path(cfg["dataset"]["train_sequences_path"])
    else:
        train_source = Path(options["data_dir"]) / "hdfs_train_xu"

    # ----------------------------
    # 2) Build model instance (benign eval)
    # ----------------------------
    Model = deeplog(
        input_size=options["input_size"],
        hidden_size=options["hidden_size"],
        num_layers=options["num_layers"],
        num_keys=options["num_classes"],
    )

    # ----------------------------
    # 3) Load sequences + weights
    # ----------------------------
    original_sequences = load_training_sequences(str(train_source))  # noqa: F821
    model_weights = _load_state_dict(ckpt_path)

    # ----------------------------
    # 4) Capacity
    # ----------------------------
    capacity_bits = count_capacity_bits(model_weights, n_bits)

    # ----------------------------
    # 5) Encoding (baseline or Huffman)
    # ----------------------------
    num_classes = int(options["num_classes"])

    if not huffman_enabled:
        binary_chunks = convert_sequences_to_binary(  # noqa: F821
            original_sequences, num_classes=num_classes, one_indexed=True
        )
  
    else:
        binary_chunks, huff_codec = convert_sequences_to_binary_huffman(  # noqa: F821
            original_sequences, num_classes=num_classes, one_indexed=True
        )
  
    
    data_bits_full = "".join(binary_chunks)
    dataset_bits_total = len(data_bits_full)

    bits_intended = min(dataset_bits_total, capacity_bits)
    data_bits_to_embed = data_bits_full[:bits_intended]

    rem = len(data_bits_to_embed) % n_bits
    padded_embed_bits = data_bits_to_embed + ("0" * (n_bits - rem) if rem else "")

    # ----------------------------
    # 6) Embed bits
    # ----------------------------
    start_encoding = time.time()
    remaining = padded_embed_bits
    new_weights = {}

    for name, param in model_weights.items():
        if remaining:
            new_param, remaining, _ = insert_data_into_tensor_all_bits(  # noqa: F821
                param, remaining, replace_last_n=n_bits
            )
        else:
            new_param = param
        new_weights[name] = new_param

    encoding_time = time.time() - start_encoding
 
    # ----------------------------
    # 7) Save attacked model (unique path + name)
    # ----------------------------
    out_dir, attacked_ckpt_path = save_attacked_model(
        new_weights=new_weights,
        cfg=cfg,
        options=options,
        n_bits=n_bits,
        approach=approach,
    )


    # ----------------------------
    # 8) Fix W&B config display (show effective values)
    # ----------------------------
    try:
        import wandb  # type: ignore
        if wandb.run is not None:
            wandb.config.update(
                {
                    "experiment.target_id": target_id,
                    "attack.params.approach": approach,
                    "attack.params.n_bits": n_bits,
                    "attack.preprocessing.huffman": bool(huffman_enabled),
                    "checkpoint.path": str(ckpt_path),
                    "paths.attacked_ckpt": str(attacked_ckpt_path),
                },
                allow_val_change=True,
            )
    except Exception:
        pass

    # ----------------------------
    # 9) Benign evaluation
    # ----------------------------
    P, R, F1, FP, FN, prediction_time = evaluate_model(  # noqa: F821
        Model, options, attacked_ckpt_path
    )

    # after saving attacked model + utility eval:
    attacker_eval = _evaluate_as_attacker(
        defended_sd=new_weights,          # IMPORTANT: attacked weights
        train_source=train_source,
        attacker_n_bits=n_bits,
        approach="baseline",
        huffman_enabled=huffman_enabled,
        options=options,
        out_dir=out_dir,
    )

    # ----------------------------
    # 11) Report
    # ----------------------------
    report: Dict[str, Any] = {
        "capacity": attacker_eval["capacity"],
        "attack": {
            "type": "lsb",
            "name": cfg["attack"].get("name", "lsb_encode"),
            "dataset": dataset_id,
            "approach": approach,
            "n_bits": n_bits,
            "huffman": bool(huffman_enabled),
            "target_id": target_id,
            "train_source": str(train_source),
            "checkpoint_path": str(ckpt_path),
            "derived_event_bit_width": attacker_eval["attack_context"]["token_bits"],
            "derived_delim_token": attacker_eval["attack_context"]["delim_token"],
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
            "decoding_time_s":  float(attacker_eval["timing"]["decoding_time_s"]),
        },
        "paths": {
            "attacked_ckpt": str(attacked_ckpt_path),
            "report_dir": str(out_dir),
        },
    }

    report_path = out_dir / (attacked_ckpt_path.stem + "__report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ----------------------------
    # 12) W&B logging (metrics)
    # ----------------------------
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

                "utility/f1": report["utility"]["f1"],
                "utility/precision": report["utility"]["precision"],
                "utility/recall": report["utility"]["recall"],

                "similarity/bit_similarity": report["similarity"]["bit_similarity"],
                "similarity/event_similarity": report["similarity"]["event_similarity"],
                "similarity/sequence_similarity_exact": report["similarity"]["sequence_similarity_exact"],
                "similarity/sequence_similarity_soft": report["similarity"]["sequence_similarity_soft"],

                "attack/approach": report["attack"]["approach"],
                "attack/n_bits": report["attack"]["n_bits"],
                "attack/huffman": int(report["attack"]["huffman"]),
            })
    except Exception:
        pass

    return report
