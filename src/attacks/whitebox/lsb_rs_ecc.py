import math
from typing import Any, Dict, List, Sequence, Tuple
import json

from src.models.deeplog import deeplog
from src.utils.lsb_utils import *
from src.defenses.whitebox.lsb_defense import _evaluate_as_attacker

def _get_nested(cfg: Dict[str, Any], path: List[str], default=None):
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _load_state_dict_any(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Checkpoint must be a state_dict or dict containing key 'state_dict'.")

def count_capacity_bits_ham(state_dict: Dict[str, torch.Tensor], n_bits: int) -> int:
    total_elems = 0
    for t in state_dict.values():
        if not torch.is_tensor(t):
            continue
        # only floats (match your insert/decode which use float_to_bin32_le)
        if not (t.dtype.is_floating_point):
            continue
        total_elems += int(t.numel())
    return total_elems * int(n_bits)
def run_lsb_rs(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    # ----------------------------
    # 0) W&B sweep overrides
    # ----------------------------
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
    approach = str(p.get("approach", "rs"))
    if approach != "rs":
        raise ValueError(f"run_lsb_rs called with approach={approach}; expected 'rs'.")

    n_bits = int(p["n_bits"])

    # pass as dict, evaluator expects dict-like rs_cfg
    rscfg_raw = p.get("rs", {}) or {}
    rs_cfg: Dict[str, Any] = {
        "k_bytes": int(rscfg_raw.get("k_bytes", 16)),
        "nsym": int(rscfg_raw.get("nsym", 32)),
        "interleave_cols": int(rscfg_raw.get("interleave_cols", 32)),
        "keep_raw_on_fail": bool(rscfg_raw.get("keep_raw_on_fail", True)),
    }
    interleave_cols = int(rs_cfg["interleave_cols"])

    dataset_id = str(_get_nested(cfg, ["dataset", "name"], options.get("dataset_id", "dataset")))
    target_id  = str(_get_nested(cfg, ["experiment", "target_id"], options.get("target_id", "target")))

    pre = cfg.get("attack", {}).get("preprocessing", {}) or {}
    huffman_enabled = bool(pre.get("huffman", False))

    # ---- paths ----
    ckpt_path = Path(cfg["checkpoint"]["path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if "dataset" in cfg and cfg["dataset"].get("train_sequences_path"):
        train_source = Path(cfg["dataset"]["train_sequences_path"])
    else:
        train_source = Path(options["data_dir"]) / "hdfs_train_xu"

    # ---- model (utility eval) ----
    Model = deeplog(
        input_size=options["input_size"],
        hidden_size=options["hidden_size"],
        num_layers=options["num_layers"],
        num_keys=options["num_classes"],
    )

    # ---- sequences + weights ----
    original_sequences = load_training_sequences(str(train_source))
    model_weights = _load_state_dict_any(ckpt_path)

    # ---- capacity ----
    capacity_bits = count_capacity_bits(model_weights, n_bits)

    # ---- base bits ----
    num_classes = int(options["num_classes"])
    w = event_bit_width(num_classes)
    delim_token = (1 << w) - 1  # not used here, but consistent

    binary_chunks_base = convert_sequences_to_binary(
        original_sequences, num_classes=num_classes, one_indexed=True
    )
    data_bits_full_base = "".join(binary_chunks_base)

    # ---- Huffman preprocessing (compressed_bits_full must match evaluator) ----
    if not huffman_enabled:
        compressed_bits_full = data_bits_full_base
    else:
        huff_chunks, _huff_codec = convert_sequences_to_binary_huffman(
            original_sequences, num_classes=num_classes, one_indexed=True
        )
        compressed_bits_full = "".join(huff_chunks)

    # ============================================================
    # Build embedding prefix EXACTLY like evaluator RS branch
    # ============================================================
    coded_bits_full, _dbg_enc = rs_fixed_block_encode_bits(compressed_bits_full, rs_cfg)
    coded_bits_total_full = len(coded_bits_full)

    # 2) Determine how much we intend to embed (aligned to interleave_cols)
    max_interleaved_len = (capacity_bits // interleave_cols) * interleave_cols
    if max_interleaved_len == 0:
        raise ValueError(f"capacity_bits={capacity_bits} too small for interleave_cols={interleave_cols}")

    coded_seg_len = min(coded_bits_total_full, max_interleaved_len)
    coded_seg = coded_bits_full[:coded_seg_len]

    interleaved_seg, _pad_interleave = interleave_bits(coded_seg, interleave_cols)
    bits_intended_embed = min(len(interleaved_seg), capacity_bits)
    bits_intended_embed = (bits_intended_embed // interleave_cols) * interleave_cols
    interleaved_to_embed = interleaved_seg[:bits_intended_embed]

    # writer pad to multiple of n_bits
    rem = len(interleaved_to_embed) % n_bits
    padded_embed_bits = interleaved_to_embed + ("0" * (n_bits - rem) if rem else "")

    # ---- embed ----
    start_encoding = time.time()
    remaining = padded_embed_bits
    new_weights = {}
    for name, param in model_weights.items():
        if remaining:
            new_param, remaining, _ = insert_data_into_tensor_all_bits(
                param, remaining, replace_last_n=n_bits
            )
        else:
            new_param = param
        new_weights[name] = new_param
    encoding_time = time.time() - start_encoding

    padded_bits_written = len(padded_embed_bits) - len(remaining)

    # ---- save attacked ----
    out_dir, attacked_ckpt_path = save_attacked_model(
        new_weights=new_weights,
        cfg=cfg,
        options=options,
        n_bits=n_bits,
        approach=approach,
    )
    out_dir = Path(out_dir)

    # ---- utility eval ----
    P, R, F1, FP, FN, prediction_time = evaluate_model(Model, options, attacked_ckpt_path)

    # ============================================================
    # Attacker evaluation (capacity/similarity/debug_rs) from evaluator
    # ============================================================
    attacker_eval = _evaluate_as_attacker(
        defended_sd=new_weights,
        train_source=train_source,
        attacker_n_bits=n_bits,
        approach="rs",
        huffman_enabled=huffman_enabled,
        rs_cfg=rs_cfg,
        options=options,
        out_dir=out_dir,
    )

    # ---- report ----
    report: Dict[str, Any] = {
        "capacity": attacker_eval["capacity"],
        "attack": {
            "type": "lsb",
            "name": cfg["attack"].get("name", "lsb_encode"),
            "dataset": dataset_id,
            "approach": "rs",
            "n_bits": int(n_bits),
            "huffman": bool(huffman_enabled),
            "target_id": target_id,
            "train_source": str(train_source),
            "checkpoint_path": str(ckpt_path),
            "derived_event_bit_width": int(attacker_eval["attack_context"]["token_bits"]),
            "derived_delim_token": int(attacker_eval["attack_context"]["delim_token"]),
            "rs": dict(rs_cfg),
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

    if "debug_rs" in attacker_eval:
        report["debug_rs"] = attacker_eval["debug_rs"]

    report_path = out_dir / (attacked_ckpt_path.stem + "__report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ---- wandb log (same keys as others) ----
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
                "attack/rs_k_bytes": rs_cfg["k_bytes"],
                "attack/rs_nsym": rs_cfg["nsym"],
                "attack/rs_interleave_cols": rs_cfg["interleave_cols"],

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