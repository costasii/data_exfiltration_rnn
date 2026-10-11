# src/attacks/lsb_redundant.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple
import json

from src.models.deeplog import deeplog
from src.utils.general_utils import seed_everything
from src.utils.lsb_utils import *

# to do: sth is wrong with capacity metrics (reporting the same number for intended redundant and capacity bits, have to make a diff)

def _load_state_dict_any(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Checkpoint must be a state_dict or dict containing key 'state_dict'.")


# def _repeat_wbit_chunks(binary_chunks: list[str], w: int, reps: int) -> str:
#     """
#     Input: list of per-sequence bitstrings where each is a concatenation of w-bit tokens
#            and ends with a w-bit delimiter token (all-ones).
#     Output: concatenated bitstream where each w-bit token is repeated `reps` times.
#     """
#     if reps < 1:
#         raise ValueError("reps must be >= 1")

#     out: list[str] = []
#     for seq_bits in binary_chunks:
#         if len(seq_bits) % w != 0:
#             raise ValueError(f"Sequence bits length {len(seq_bits)} not divisible by w={w}")
#         # split into w-bit tokens and repeat each token reps times
#         for i in range(0, len(seq_bits), w):
#             tok = seq_bits[i : i + w]
#             if len(tok) != w or any(c not in "01" for c in tok):
#                 raise ValueError(f"Expected {w}-bit token, got: {tok!r}")
#             out.extend([tok] * reps)
#     return "".join(out)


# def _majority_vote_repeated_bits(
#     bitstream: str, token_bits: int, reps: int
# ) -> tuple[str, dict]:
#     """
#     Majority vote per bit across `reps` repeated tokens of width `token_bits`.
#     Returns recovered bitstream in base token space (multiple of token_bits),
#     plus debug diagnostics.
#     """
#     group_bits = token_bits * reps
#     n_groups = len(bitstream) // group_bits

#     recovered: list[str] = []
#     tokens_with_any_disagreement = 0
#     total_bit_disagreements = 0  # bit positions where not all reps match

#     for gi in range(n_groups):
#         base = gi * group_bits
#         copies = [
#             bitstream[base + r * token_bits : base + (r + 1) * token_bits]
#             for r in range(reps)
#         ]

#         voted = []
#         any_disagreement_this_token = False
#         for j in range(token_bits):
#             ones = sum(1 for r in range(reps) if copies[r][j] == "1")
#             maj = "1" if ones >= (reps // 2 + 1) else "0"
#             voted.append(maj)

#             # disagreement if not all same across reps
#             first = copies[0][j]
#             if any(copies[r][j] != first for r in range(1, reps)):
#                 any_disagreement_this_token = True
#                 total_bit_disagreements += 1

#         if any_disagreement_this_token:
#             tokens_with_any_disagreement += 1

#         recovered.append("".join(voted))

#     diagnostics = {
#         "token_bits": int(token_bits),
#         "reps": int(reps),
#         "groups_total": int(n_groups),
#         "tokens_with_any_disagreement": int(tokens_with_any_disagreement),
#         "total_bit_disagreements": int(total_bit_disagreements),
#         "dropped_tail_bits": int(len(bitstream) - n_groups * group_bits),
#     }
#     return "".join(recovered), diagnostics
def run_lsb_redundant(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    # ----------------------------
    # 0) W&B sweep overrides (same pattern as baseline)
    # ----------------------------
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

    # ----------------------------
    # Params 
    # ----------------------------
    p = cfg["attack"]["params"]
    approach = str(p.get("approach", "redundant"))
    if approach != "redundant":
        raise ValueError(f"run_lsb_redundant called with approach={approach}; expected 'redundant'.")

    n_bits = int(p["n_bits"])

    # redundant-only params
    reps = int(p["reps"])
    block_bits = int(p.get("redundancy_block_bits", 1024))

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
    # 1) Resolve paths (same as baseline)
    # ----------------------------
    ckpt_path = Path(cfg["checkpoint"]["path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if "dataset" in cfg and cfg["dataset"].get("train_sequences_path"):
        train_source = Path(cfg["dataset"]["train_sequences_path"])
    else:
        train_source = Path(options["data_dir"]) / "hdfs_train_xu"


    # ---- 1) Build model instance (for benign eval) ----
    Model = deeplog(
        input_size=options["input_size"],
        hidden_size=options["hidden_size"],
        num_layers=options["num_layers"],
        num_keys=options["num_classes"],
    )

    # ---- 2) Load sequences ----
    original_sequences = load_training_sequences(str(train_source))  # noqa: F821

    # ---- 3) Load checkpoint weights + capacity ----
    model_weights = _load_state_dict_any(ckpt_path)
    capacity_bits = count_capacity_bits(model_weights, n_bits)

    # ---- 4) Encode baseline stream (for comparison + metrics) ----
    num_classes = int(options["num_classes"])
    w = event_bit_width(num_classes)  # noqa: F821
    delim_token = (1 << w) - 1  
    base_chunks = convert_sequences_to_binary(
        original_sequences, num_classes=options["num_classes"], one_indexed=True
    )
    base_bits_full = "".join(base_chunks)
    dataset_bits_total_base = len(base_bits_full)

    # ---- 5) Preprocess (Huffman optional): produce compressed_bits_full ----
    # Define compression metrics exactly like baseline expects (always defined)
    if not huffman_enabled:
        compressed_bits_full = base_bits_full
        huff_codec = None
        huff_chunks = None

        dataset_bits_total_compressed = dataset_bits_total_base
        compression_ratio = 1.0
        compression_savings_pct = 0.0
        effective_capacity_gain_x = 1.0
        effective_capacity_gain_pct = 0.0
    else:
        huff_chunks, huff_codec = convert_sequences_to_binary_huffman(
            original_sequences, num_classes=options["num_classes"], one_indexed=True
        )
        compressed_bits_full = "".join(huff_chunks)
        dataset_bits_total_compressed = len(compressed_bits_full)

        if dataset_bits_total_base > 0 and dataset_bits_total_compressed > 0:
            compression_ratio = dataset_bits_total_compressed / dataset_bits_total_base
            compression_savings_pct = (1.0 - compression_ratio) * 100.0
            effective_capacity_gain_x = dataset_bits_total_base / dataset_bits_total_compressed
            effective_capacity_gain_pct = (float(effective_capacity_gain_x) - 1.0) * 100.0
        else:
            # degenerate case
            compression_ratio = None
            compression_savings_pct = None
            effective_capacity_gain_x = None
            effective_capacity_gain_pct = None

    # ---- 6) Apply redundancy to COMPRESSED bitstream (block repetition) ----
    redundant_bits_full, rep_dbg = repeat_bit_blocks(
        compressed_bits_full, block_bits=block_bits, reps=reps
    )
    dataset_bits_total_redundant = len(redundant_bits_full)

    # How many redundant bits can we embed?
    bits_intended_redundant = min(dataset_bits_total_redundant, capacity_bits)

    # Unpadded redundant bits we intend to represent
    data_bits_to_embed_redundant = redundant_bits_full[:bits_intended_redundant]

    # Pad to multiple of n_bits for writer requirement
    rem = len(data_bits_to_embed_redundant) % n_bits
    padded_embed_bits = data_bits_to_embed_redundant + ("0" * (n_bits - rem) if rem else "")

    # ---- 7) Embed ----
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

    # ---- 8) Save attacked model (same signature style as baseline) ----
    out_dir, attacked_ckpt_path = save_attacked_model(
        new_weights=new_weights,
        cfg=cfg,
        options=options,
        n_bits=n_bits,
        approach=approach,
    )  # noqa: F821
    out_dir = Path(out_dir)

    # ---- 9) Benign evaluation ----
    P, R, F1, FP, FN, prediction_time = evaluate_model(Model, options, attacked_ckpt_path)  # noqa: F821

    # ---- 10) Decode + majority vote back to COMPRESSED bitstream ----
    start_decoding = time.time()
    padded_bits_written = len(padded_embed_bits) - len(remaining)
    exfil_padded = decode_all_parameters(new_weights, n_bits, padded_bits_written)  # noqa: F821
    decoding_time = time.time() - start_decoding

    # drop padding (only compare what we intended as *data*)
    exfil_bits_redundant = exfil_padded[:len(data_bits_to_embed_redundant)]

    recovered_compressed_bits_padded, vote_dbg = majority_vote_bit_blocks(
        exfil_bits_redundant, block_bits=block_bits, reps=reps
    )

    # Align compare length to full recovered blocks ONLY (no headers, no guesswork)
    group_bits = block_bits * reps
    groups_embedded = len(data_bits_to_embed_redundant) // group_bits
    recovered_len_padded = groups_embedded * block_bits

    recovered_compressed_bits_padded = recovered_compressed_bits_padded[:recovered_len_padded]
    orig_compressed_bits_cmp = compressed_bits_full[:recovered_len_padded]

    # ---- capacity metrics (baseline-aligned) ----
    # 1) how many COMPRESSED bits were intended given redundancy overhead
    compressed_bits_intended = min(dataset_bits_total_compressed, groups_embedded * block_bits)

    # 2) convert intended COMPRESSED budget -> intended BASE bits
    if not huffman_enabled:
        base_bits_intended = min(dataset_bits_total_base, compressed_bits_intended)
    else:
        base_bits_intended = base_bits_intended_from_compressed_budget(
            base_chunks=base_chunks,
            comp_chunks=huff_chunks,
            comp_bits_budget=compressed_bits_intended,
        )

    # ---- similarity metrics in COMPRESSED-bit space (after vote) ----
    bit_sim, bit_errs, bit_compared = bit_similarity(
        recovered_compressed_bits_padded, orig_compressed_bits_cmp
    )  # noqa: F821

    # ---- parse to events/sequences (depends on preprocessing) ----
    if not huffman_enabled:
        orig_events_all = bits_to_events(orig_compressed_bits_cmp, chunk_bits=w)  # noqa: F821
        dec_events_all  = bits_to_events(recovered_compressed_bits_padded, chunk_bits=w)  # noqa: F821
    else:
        orig_events_all, _ = huffman_bits_to_events(orig_compressed_bits_cmp, huff_codec, delim_token)  # noqa: F821
        dec_events_all, _  = huffman_bits_to_events(recovered_compressed_bits_padded, huff_codec, delim_token)  # noqa: F821

    orig_seqs_decodable = events_to_sequences(orig_events_all, delim_token=delim_token)  # noqa: F821
    dec_seqs_decodable  = events_to_sequences(dec_events_all,  delim_token=delim_token)  # noqa: F821
    orig_seqs_cmp, dec_seqs_cmp = truncate_to_full_sequences(orig_seqs_decodable, dec_seqs_decodable)  # noqa: F821

    seq_sim, seq_correct, seq_total = sequence_similarity(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    evt_sim, evt_correct, evt_total = event_similarity(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    mean_nhd, _ = mean_normalized_hamming_distance(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    seq_recovery_soft = 1.0 - mean_nhd

    evt_membership = event_membership_rate(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    seq_membership = sequence_membership_rate(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821

    seq_sim_weighted = seq_similarity_weighted_by_len(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    seq_sim_unique = sequence_similarity_unique(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    seq_membership_unique = sequence_membership_rate_unique(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821

    evt_sim_weighted = evt_sim


    # ---- Baseline-equivalent capacity fields ----
    attack_capacity = int(capacity_bits)

    dataset_bits_total = int(dataset_bits_total_base)     # baseline's dataset_bits_total is BASE space
    bits_intended = int(base_bits_intended)              # baseline key: bits_intended (BASE)
    capacity_coverage = (bits_intended / dataset_bits_total) if dataset_bits_total else 0.0

    bits_baseline_full = int(dataset_bits_total_base)
    bits_preproc_full  = int(dataset_bits_total_compressed)
    preproc_enabled = bool(huffman_enabled)

    # ---- Report ----
    report: Dict[str, Any] = {
        "capacity": {
            "attack_capacity": int(attack_capacity),
            "dataset_bits_total": int(dataset_bits_total),
            "bits_intended": int(bits_intended),
            "coverage": float(capacity_coverage),
            "padded_bits_written": int(padded_bits_written),

            "preproc_enabled": bool(preproc_enabled),
            "bits_baseline_full": int(bits_baseline_full),
            "bits_preproc_full": int(bits_preproc_full),

            "compression_ratio": compression_ratio,
            "compression_savings_pct": compression_savings_pct,
            "effective_capacity_gain_x": effective_capacity_gain_x,
            "effective_capacity_gain_pct": effective_capacity_gain_pct,
        },
        "attack": {
            "type": "lsb",
            "name": cfg["attack"].get("name", "lsb_encode"),
            "dataset": dataset_id,

            "approach": approach,
            "n_bits": int(n_bits),
            "target_id": target_id,

            "preprocessing": {"huffman": bool(huffman_enabled)},
            "redundancy": {"reps": int(reps), "block_bits": int(block_bits)},
            "train_source": str(train_source),
            "checkpoint_path": str(ckpt_path),
            "derived_event_bit_width": int(w),
            "derived_delim_token": int(delim_token),
        },
        "utility": {
            "precision": float(P),
            "recall": float(R),
            "f1": float(F1),
            "false_positives": int(FP),
            "false_negatives": int(FN),
        },
        "similarity": {
            "bit_similarity": float(bit_sim),
            "bit_errors": int(bit_errs),
            "bits_compared": int(bit_compared),

            "event_similarity": float(evt_sim),
            "events_correct": int(evt_correct),
            "events_compared": int(evt_total),

            "sequence_similarity_exact": float(seq_sim),
            "sequences_correct": int(seq_correct),
            "sequences_compared": int(seq_total),

            "sequence_similarity_soft": float(seq_recovery_soft),
            "event_membership_rate": float(evt_membership),
            "sequence_membership_rate": float(seq_membership),

            "event_similarity_weighted": float(evt_sim_weighted),
            "sequence_similarity_weighted": float(seq_sim_weighted),

            "sequence_similarity_unique": float(seq_sim_unique),
            "sequence_membership_rate_unique": float(seq_membership_unique),

      

            "sequences_decoded_full": int(len(dec_seqs_decodable)),
            "sequences_original_full_in_encoded_portion": int(len(orig_seqs_decodable)),
        },
        "debug_redundancy": {
            **rep_dbg,
            **vote_dbg,
            "groups_embedded": int(groups_embedded),
            "decoded_redundant_bits_used": int(len(exfil_bits_redundant)),
        },
        "timing": {
            "encoding_time_s": float(encoding_time),
            "prediction_time_s": float(prediction_time),
            "decoding_time_s": float(decoding_time),
        },
        "paths": {
            "attacked_ckpt": str(attacked_ckpt_path),
            "report_dir": str(out_dir),
        },
        "redundancy": {
            "reps": int(reps),
            "block_bits": int(block_bits),
            "dataset_bits_total_redundant": int(dataset_bits_total_redundant),
            "bits_intended_redundant": int(bits_intended_redundant),
            "groups_embedded": int(groups_embedded),
        },
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- W&B logging (aligned with baseline keys + redundancy extras) ----
    try:
        import wandb  # type: ignore
        if wandb.run is not None:
            wandb.log({
                "capacity/attack_capacity": report["capacity"]["attack_capacity"],
                "capacity/dataset_bits_total": report["capacity"]["dataset_bits_total"],
                "capacity/bits_intended": report["capacity"]["bits_intended"],
                "capacity/coverage": report["capacity"]["coverage"],
                "capacity/padded_bits_written": report["capacity"]["padded_bits_written"],
                "capacity/compression_enabled": int(huffman_enabled),
                "capacity/compression_savings_pct": report["capacity"]["compression_savings_pct"],
                "capacity/effective_capacity_gain_x": report["capacity"]["effective_capacity_gain_x"],

                "attack/approach": approach,
                "attack/n_bits": int(n_bits),
                "attack/dataset": dataset_id,
                "attack/target_id": target_id,

                "attack/preproc_huffman": int(huffman_enabled),
                "attack/attacked_ckpt": str(attacked_ckpt_path),


                "utility/precision": float(P),
                "utility/recall": float(R),
                "utility/f1": float(F1),
                "utility/fp": int(FP),
                "utility/fn": int(FN),

                "similarity/bit_similarity": report["similarity"]["bit_similarity"],
                "similarity/event_similarity": report["similarity"]["event_similarity"],
                "similarity/sequence_similarity_exact": report["similarity"]["sequence_similarity_exact"],
                "similarity/sequence_similarity_soft": report["similarity"]["sequence_similarity_soft"],

                "diagnostics/sequences_decoded_full": report["similarity"]["sequences_decoded_full"],

                "timing/encoding_time_s": float(encoding_time),
                "timing/prediction_time_s": float(prediction_time),
                "timing/decoding_time_s": float(decoding_time),

                # redundancy extras
                "redundancy/reps": int(reps),
                "redundancy/block_bits": int(block_bits),
                "redundancy/dataset_bits_total_redundant": int(dataset_bits_total_redundant),
                "redundancy/bits_intended_redundant": int(bits_intended_redundant),
            })
    except Exception:
        pass

    return report
