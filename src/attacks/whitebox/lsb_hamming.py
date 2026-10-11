import math
from typing import Any, Dict, List, Sequence, Tuple
import json

from src.models.deeplog import deeplog
from src.utils.general_utils import seed_everything
from src.utils.lsb_utils import *

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

def run_lsb_hamming(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hamming(15,11) + interleaving LSB attack (white-box):

      1) Base-encode ALL training sequences as w-bit tokens + delimiter (delimiter = all-ones token)
      2) Hamming(15,11) encode the base bitstream (bit-level ECC)
      3) Interleave encoded bits (transpose with cols=32 by default)
      4) Embed sequentially into checkpoint by replacing last n_bits of each float
      5) Save attacked checkpoint
      6) Evaluate benign task (DeepLog)
      7) Decode bits, deinterleave, Hamming-decode back to BASE bits
      8) Same similarity + membership metrics as baseline/redundant
    """
    p = cfg["attack"]["params"]
    approach = str(p.get("approach", "hamming"))
    if approach != "hamming":
        raise ValueError(f"run_lsb_hamming called with approach={approach}; expected 'hamming'.")

    n_bits = int(p["n_bits"])
    hcfg = p.get("hamming", {}) or {}
    interleave_cols = int(hcfg.get("interleave_cols", 32))

    # ---- 0) Resolve paths ----
    ckpt_path = Path(cfg["checkpoint"]["path"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if "dataset" in cfg and cfg["dataset"].get("train_sequences_path"):
        train_source = Path(cfg["dataset"]["train_sequences_path"])
    else:
        train_source = Path(options["data_dir"]) / "hdfs_train_xu"

    # ---- 1) Build model instance (for benign eval) ----
    Model = deeplog(  # noqa: F821
        input_size=options["input_size"],
        hidden_size=options["hidden_size"],
        num_layers=options["num_layers"],
        num_keys=options["num_classes"],
    )

    # ---- 2) Load sequences to encode ----
    original_sequences = load_training_sequences(str(train_source))  # noqa: F821

    # ---- 3) Load checkpoint weights ----
    model_weights = _load_state_dict_any(ckpt_path)  # noqa: F821
    capacity_bits = count_capacity_bits_ham(model_weights, n_bits)

    # ---- 4) Base encode dataset (w-bit tokens + delimiter) ----
    num_classes = int(options["num_classes"])
    w = event_bit_width(num_classes)  # noqa: F821
    delim_token = (1 << w) - 1

    binary_chunks_base = convert_sequences_to_binary(  # noqa: F821
        original_sequences, num_classes=num_classes, one_indexed=True
    )
    base_bits_full = "".join(binary_chunks_base)

    # ---- 5) Hamming encode (15,11) ----
    coded_full, dbg_enc = hamming15_11_encode_bits(base_bits_full)
    dataset_bits_total_coded = len(coded_full)

    # IMPORTANT: interleaving pads to multiple of interleave_cols
    # So we must budget capacity in INTERLEAVED space, not coded space.
    max_interleaved_len = (capacity_bits // interleave_cols) * interleave_cols
    if max_interleaved_len == 0:
        raise ValueError(f"capacity_bits={capacity_bits} too small for interleave_cols={interleave_cols}")

    # We will embed at most max_interleaved_len interleaved bits.
    # That corresponds to at most max_interleaved_len coded bits, but coded bits may be shorter by pad_interleave later.
    coded_seg_len = min(dataset_bits_total_coded, max_interleaved_len)
    coded_seg = coded_full[:coded_seg_len]
    dataset_bits_total = len(base_bits_full)

    base_bits_intended = (len(coded_seg) // 15) * 11
    base_bits_intended = min(base_bits_intended, dataset_bits_total)

    capacity_coverage = (
        base_bits_intended / dataset_bits_total
        if dataset_bits_total else 0.0
    )


    # Interleave intended coded segment (this will pad to exactly a multiple of interleave_cols)
    interleaved_seg, pad_interleave = interleave_bits(coded_seg, interleave_cols)
    # By construction this must fit.
    assert len(interleaved_seg) <= capacity_bits

    # Pad to multiple of n_bits for writer requirement
    rem = len(interleaved_seg) % n_bits
    padded_embed_bits = interleaved_seg + ("0" * (n_bits - rem) if rem else "")

    # --- embed padded_embed_bits sequentially ---
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

    # Compute how many bits were ACTUALLY written
    bits_written = len(padded_embed_bits) - len(remaining)

    # We only evaluate on the portion that was actually written.
    # This is the key fix.
    interleaved_written = interleaved_seg[:bits_written]
    bits_intended = len(interleaved_written)


    # ---- 6) Save attacked model ----
    out_dir, attacked_ckpt_path = save_attacked_model(  # noqa: F821
        new_weights, cfg, options, n_bits, approach
    )
    out_dir = Path(out_dir)

    # ---- 7) Benign evaluation ----
    P, R, F1, FP, FN, prediction_time = evaluate_model(  # noqa: F821
         Model, options, attacked_ckpt_path
    )

    # ---- 8) Decode -> deinterleave -> Hamming decode to BASE ----
    start_decoding = time.time()

    # Decode exactly the number of bits that were written
    exfil_interleaved = decode_all_parameters(  # noqa: F821
        new_weights,
        bit_len=n_bits,
        bit_target_len=bits_written,
    )

    decoding_time = time.time() - start_decoding

    # Now compare only the written part (these should be identical without defense)
    exfil_interleaved = exfil_interleaved[:len(interleaved_written)]
    assert len(exfil_interleaved) == len(interleaved_written)

    # Deinterleave requires length divisible by cols.
    # If we are cut off mid-block, we must drop the tail to a multiple of cols.
    L_block = (len(exfil_interleaved) // interleave_cols) * interleave_cols
    exfil_interleaved_blk = exfil_interleaved[:L_block]
    interleaved_written_blk = interleaved_written[:L_block]

    # Optional sanity check (no-defense should be 1.0)
    # sim_inter, _, _ = bit_similarity(exfil_interleaved_blk, interleaved_written_blk)

    exfil_coded_padded = deinterleave_bits(exfil_interleaved_blk, cols=interleave_cols)

    # Now we are back in coded space, but remember:
    # pad_interleave was computed for the FULL interleaved_seg.
    # For a truncated block L_block, the corresponding coded length is:
    # coded_len_blk = L_block - pad_for_this_block
    # easiest: just de-pad only if we still have the full last padded row.
    # Safe approach: do NOT remove pad_interleave here; instead decode full 15-bit blocks only.

    # Keep only full 15-bit blocks for BCH/Hamming decoding
    coded_for_decode = exfil_coded_padded[: (len(exfil_coded_padded) // 15) * 15]

    max_plain_bits_decodable = (len(coded_for_decode) // 15) * 11
    plain_target_len = min(len(base_bits_full), max_plain_bits_decodable)

    recovered_base_bits, dbg_dec = hamming15_11_decode_bits(
        coded_for_decode,
        plain_bits_len=plain_target_len,
    )

    orig_base_bits_cmp = base_bits_full[:len(recovered_base_bits)]


    # ---- similarity metrics (BASE space) ----
    bit_sim, bit_errs, bit_compared = bit_similarity(recovered_base_bits, orig_base_bits_cmp)  # noqa: F821

    orig_events_all = bits_to_events(orig_base_bits_cmp, chunk_bits=w)  # noqa: F821
    dec_events_all  = bits_to_events(recovered_base_bits, chunk_bits=w)  # noqa: F821

    orig_seqs_decodable = events_to_sequences(orig_events_all, delim_token=delim_token)  # noqa: F821
    dec_seqs_decodable  = events_to_sequences(dec_events_all,  delim_token=delim_token)  # noqa: F821

    orig_seqs_cmp, dec_seqs_cmp = truncate_to_full_sequences(orig_seqs_decodable, dec_seqs_decodable)  # noqa: F821

    seq_sim, seq_correct, seq_total = sequence_similarity(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    evt_sim, evt_correct, evt_total = event_similarity(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821

    mean_nhd, _ = mean_normalized_hamming_distance(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    seq_recovery_soft = 1.0 - mean_nhd

    # unified extras (same as defense)
    evt_membership = event_membership_rate(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    seq_membership = sequence_membership_rate(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    seq_sim_weighted = seq_similarity_weighted_by_len(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    seq_sim_unique = sequence_similarity_unique(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    seq_membership_unique = sequence_membership_rate_unique(orig_seqs_cmp, dec_seqs_cmp)  # noqa: F821
    evt_sim_weighted = evt_sim

    # ---- 9) Report ----
    report: Dict[str, Any] = {
        "capacity": {
            "attack_capacity": int(capacity_bits),
            "dataset_bits_total": int(dataset_bits_total_coded),
            "bits_intended": int(base_bits_intended),
            "coverage": float(capacity_coverage),
            "leftover_bits_after_embedding": int(len(remaining)),
        },
        "attack": {
            "type": "lsb",
            "name": cfg["attack"].get("name", "lsb_encode"),
            "approach": approach,
            "n_bits": n_bits,
            "train_source": str(train_source),
            "checkpoint_path": str(ckpt_path),
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

            "sequences_decoded_full": int(len(dec_seqs_decodable)),
            "sequences_original_full_in_encoded_portion": int(len(orig_seqs_decodable)),

            # unified extras
            "event_membership_rate": float(evt_membership),
            "sequence_membership_rate": float(seq_membership),
            "event_similarity_weighted": float(evt_sim_weighted),
            "sequence_similarity_weighted": float(seq_sim_weighted),
            "sequence_similarity_unique": float(seq_sim_unique),
            "sequence_membership_rate_unique": float(seq_membership_unique),
        },
        "debug_hamming": {
            "hamming": {"code_n": 15, "data_k": 11},
            "interleave_cols": int(interleave_cols),
            "interleave_pad_bits": int(pad_interleave),
            "encode": dbg_enc,
            "decode": dbg_dec,
            "coded_bits_total_full": int(dataset_bits_total_coded),
            "coded_bits_used": int(len(coded_seg)),
            "interleaved_bits_used": int(len(interleaved_seg)),
            "decoded_base_bits_len": int(len(recovered_base_bits)),
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
    }

    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---- 10) Optional: W&B logging (same keys as other attacks; debug optional) ----
    try:
        import wandb  # type: ignore
        if wandb.run is not None:
            sim = report["similarity"]
            cap = report["capacity"]
            wandb.log({
                "capacity/attack_capacity": cap["attack_capacity"],
                "capacity/dataset_bits_total": cap["dataset_bits_total"],
                "capacity/bits_intended": cap["bits_intended"],
                "capacity/coverage": cap["coverage"],
                "capacity/leftover_bits_after_embedding": cap["leftover_bits_after_embedding"],

                "attack/name": report["attack"]["name"],
                "attack/approach": approach,
                "attack/n_bits": n_bits,

                "utility/precision": report["utility"]["precision"],
                "utility/recall": report["utility"]["recall"],
                "utility/f1": report["utility"]["f1"],
                "utility/fp": report["utility"]["false_positives"],
                "utility/fn": report["utility"]["false_negatives"],

                "similarity/bit_similarity": sim["bit_similarity"],
                "similarity/bit_errors": sim["bit_errors"],
                "similarity/bits_compared": sim["bits_compared"],

                "similarity/event_similarity": sim["event_similarity"],
                "similarity/sequence_similarity_exact": sim["sequence_similarity_exact"],
                "similarity/sequence_similarity_soft": sim["sequence_similarity_soft"],

                "similarity/event_membership_rate": sim["event_membership_rate"],
                "similarity/sequence_membership_rate": sim["sequence_membership_rate"],
                "similarity/sequence_similarity_weighted": sim["sequence_similarity_weighted"],
                "similarity/sequence_similarity_unique": sim["sequence_similarity_unique"],
                "similarity/sequence_membership_rate_unique": sim["sequence_membership_rate_unique"],

                "timing/encoding_time_s": report["timing"]["encoding_time_s"],
                "timing/prediction_time_s": report["timing"]["prediction_time_s"],
                "timing/decoding_time_s": report["timing"]["decoding_time_s"],
            })
    except Exception:
        pass

    return report
