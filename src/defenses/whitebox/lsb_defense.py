import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Sequence, Tuple, List, Optional

import torch

from typing import Dict, Tuple, List

from src.models.deeplog import deeplog

from src.utils.lsb_utils import *

from src.utils.lsb_utils import _ensure_dir
from src.utils.lsb_utils import _rand_bits
from src.utils.lsb_utils import _repeat_fixed_wbit_tokens_from_bits
from src.utils.lsb_utils import _majority_vote_fixed_tokens, _majority_vote_codewords, _build_huffman_codebook, _huffman_encode_symbols, _huffman_decode_bits, _repeat_codewords, _prefix_by_budget_codewords


def _load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    """Support both raw state_dict and {'state_dict': ...} checkpoints."""
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint must be a state_dict or dict containing key 'state_dict'.")
    return obj


# ============================================================
# Main evaluation function
# ============================================================
def _evaluate_as_attacker(
    *,
    defended_sd: Dict[str, "torch.Tensor"],
    train_source: Path,
    attacker_n_bits: int,
    approach: str,  # "baseline" | "redundant" | "repetition" | "hamming" | "rs"
    options: Dict[str, Any],
    out_dir: Path,

    # defaults for non-baseline approaches
    token_repeat: int = 1,
    redundancy_block_bits: int = 32,
    huffman_enabled: bool = False,
    repetition_reps: int = 1,
    hamming_interleave_cols: int = 32,
    rs_cfg: Dict[str, Any] | None = None,
    debug_k: int = 2,
) -> Dict[str, Any]:
    payload_is_huffman_bits = False

    rs_cfg = rs_cfg or {}
    """
    Evaluates exfiltration AFTER defense, from the attacker's perspective.
    Only difference between approaches is encode/decode path.
    Everything else matches the attack pipeline metrics.
    """
    num_classes = int(options["num_classes"])
    w = event_bit_width(num_classes)
    delim_token = (1 << w) - 1

    # ---- 1) Load original sequences ----
    original_sequences = load_training_sequences(str(train_source))  # noqa: F821

    total_events_in_dataset = sum(len(s) for s in original_sequences)  # events only, no delimiter
    
    # ---- 2) BASE encoding (fixed width, for reference/metrics) ----
    binary_chunks_base = convert_sequences_to_binary(
        original_sequences,
        num_classes=num_classes,
        one_indexed=True,
    )
    data_bits_full_base = "".join(binary_chunks_base)
    dataset_bits_total_base = len(data_bits_full_base)

    # ---- 2b) OPTIONAL preprocessing: Huffman compress ----
    huff_codec = None
    huff_chunks = None

    if not huffman_enabled:
        compressed_bits_full = data_bits_full_base
        dataset_bits_total_compressed = dataset_bits_total_base
        compression_ratio = 1.0
        compression_savings_pct = 0.0
        effective_capacity_gain_x = 1.0
        effective_capacity_gain_pct = 0.0
    else:
        huff_chunks, huff_codec = convert_sequences_to_binary_huffman(
            original_sequences,
            num_classes=num_classes,
            one_indexed=True,
        )
        compressed_bits_full = "".join(huff_chunks)
        dataset_bits_total_compressed = len(compressed_bits_full)

        if dataset_bits_total_base > 0 and dataset_bits_total_compressed > 0:
            compression_ratio = dataset_bits_total_compressed / dataset_bits_total_base
            compression_savings_pct = (1.0 - compression_ratio) * 100.0
            effective_capacity_gain_x = dataset_bits_total_base / dataset_bits_total_compressed
            effective_capacity_gain_pct = (float(effective_capacity_gain_x) - 1.0) * 100.0
        else:
            compression_ratio = None
            compression_savings_pct = None
            effective_capacity_gain_x = None
            effective_capacity_gain_pct = None

    # ---- 3) Capacity in model ----
    capacity_bits = count_capacity_bits(defended_sd, attacker_n_bits)  # noqa: F821

    redundancy_dbg = None
    vote_dbg = None
    decoded_redundant_bits_used = None
    hamming_dbg = None
    rs_dbg = None

    # outputs of branch:
    #   recovered_compressed_bits
    #   orig_compressed_bits_cmp
    #   dataset_bits_total (embedding-space dataset size for coverage)
    #   bits_intended (embedding-space intended bits)
    #   padded_bits_written, padded_embed_bits, decoding_time
    # ============================================================
    # APPROACH A: BASELINE (raw COMPRESSED bits embedded)
    # ============================================================
    if approach == "baseline":
        payload_is_huffman_bits = bool(huffman_enabled)   # baseline compares in compressed space
        compressed_bits_intended = min(dataset_bits_total_compressed, capacity_bits)
        data_bits_to_embed = compressed_bits_full[:compressed_bits_intended]

        rem = len(data_bits_to_embed) % attacker_n_bits
        padded_embed_bits = data_bits_to_embed + ("0" * (attacker_n_bits - rem) if rem else "")

        padded_bits_written = min(len(padded_embed_bits), (capacity_bits // attacker_n_bits) * attacker_n_bits)

        start_decoding = time.time()
        exfil_padded = decode_all_parameters(defended_sd, attacker_n_bits, padded_bits_written)  # noqa: F821
        decoding_time = time.time() - start_decoding

        exfil_bits_raw = exfil_padded[:len(data_bits_to_embed)]

        recovered_compressed_bits = exfil_bits_raw
        orig_compressed_bits_cmp = data_bits_to_embed

        dataset_bits_total_rep_full = dataset_bits_total_compressed  # embedding is compressed bits
        bits_intended_rep = len(data_bits_to_embed)

        payload_bits_intended = orig_compressed_bits_cmp
        payload_bits_decoded = recovered_compressed_bits

    # ============================================================
    # APPROACH B: REDUNDANT (repeat bit-blocks in COMPRESSED space, then majority vote)
    # ============================================================
    elif approach == "redundant":
        reps = int(token_repeat)
        block_bits = int(redundancy_block_bits)

        redundant_full, rep_dbg = repeat_bit_blocks(  # noqa: F821
            compressed_bits_full,
            block_bits=block_bits,
            reps=reps,
        )
        dataset_bits_total = len(redundant_full)

        bits_intended = min(dataset_bits_total, capacity_bits)
        data_bits_to_embed = redundant_full[:bits_intended]

        rem = len(data_bits_to_embed) % attacker_n_bits
        padded_embed_bits = data_bits_to_embed + ("0" * (attacker_n_bits - rem) if rem else "")

        padded_bits_written = min(len(padded_embed_bits), (capacity_bits // attacker_n_bits) * attacker_n_bits)

        start_decoding = time.time()
        exfil_padded = decode_all_parameters(defended_sd, attacker_n_bits, padded_bits_written)  # noqa: F821
        decoding_time = time.time() - start_decoding

        exfil_bits_raw = exfil_padded[:len(data_bits_to_embed)]
        decoded_redundant_bits_used = len(exfil_bits_raw)

        recovered_padded, vote_dbg = majority_vote_bit_blocks(  # noqa: F821
            exfil_bits_raw,
            block_bits=block_bits,
            reps=reps,
        )

        # align to full recovered blocks (no headers)
        group_bits = block_bits * reps
        groups_embedded = len(exfil_bits_raw) // group_bits
        recovered_len = groups_embedded * block_bits
        compressed_bits_intended = min(dataset_bits_total_compressed, groups_embedded * block_bits)


        recovered_compressed_bits = recovered_padded[:recovered_len]
        orig_compressed_bits_cmp = compressed_bits_full[:recovered_len]

        redundancy_dbg = {
            "reps": int(reps),
            "block_bits": int(block_bits),
            **(rep_dbg or {}),
            "groups_embedded": int(groups_embedded),
        }



    elif approach == "repetition":
        reps = int(repetition_reps)

        # Compare in BASE payload space for BOTH modes
        payload_is_huffman_bits = False

        if not huffman_enabled:
            base_bits = data_bits_full_base[: (len(data_bits_full_base) // w) * w]
            repeated_full = _repeat_fixed_wbit_tokens_from_bits(base_bits, w=w, reps=reps)

            dataset_bits_total_rep_full = len(repeated_full)

            group_bits = w * reps
            groups_by_capacity = capacity_bits // group_bits

            n_tokens_total = len(base_bits) // w
            groups_embedded = min(int(groups_by_capacity), int(n_tokens_total))

            bits_intended_rep = groups_embedded * group_bits
            data_bits_to_embed = repeated_full[:bits_intended_rep]

            # ---- decode from model (embedding space) ----
            rem = len(data_bits_to_embed) % attacker_n_bits
            padded_embed_bits = data_bits_to_embed + ("0" * (attacker_n_bits - rem) if rem else "")
            padded_bits_written = min(len(padded_embed_bits), (capacity_bits // attacker_n_bits) * attacker_n_bits)

            start_decoding = time.time()
            exfil_padded = decode_all_parameters(defended_sd, attacker_n_bits, padded_bits_written)
            decoding_time = time.time() - start_decoding

            exfil_bits_raw = exfil_padded[:len(data_bits_to_embed)]
            decoded_redundant_bits_used = len(exfil_bits_raw)

            voted_bits, vote_dbg = _majority_vote_fixed_tokens(exfil_bits_raw, token_bits=w, reps=reps)

            payload_bits_intended_len = groups_embedded * w
            payload_bits_intended = base_bits[:payload_bits_intended_len]
            payload_bits_decoded  = voted_bits[:payload_bits_intended_len]

            redundancy_dbg = {
                "reps": int(reps),
                "unit": "token_wbit",
                "groups_embedded": int(groups_embedded),
                **(vote_dbg or {}),
                "decoded_redundant_bits_used": int(decoded_redundant_bits_used),
            }

        else:
            base_bits = data_bits_full_base[: (len(data_bits_full_base) // w) * w]
            tokens_all = bits_to_events(base_bits, chunk_bits=w)

            codebook = _build_huffman_codebook(tokens_all)
            _bits_huff_full, codewords_all = _huffman_encode_symbols(tokens_all, codebook)

            repeated_full = _repeat_codewords(codewords_all, reps=reps)
            dataset_bits_total_rep_full = len(repeated_full)

            bit_budget = min(dataset_bits_total_rep_full, capacity_bits)
            codewords_prefix, _used_bits = _prefix_by_budget_codewords(
                codewords_all, reps=reps, bit_budget=bit_budget
            )

            # CAP by dataset symbol count (safety, ensures coverage<=1)
            n_syms_total = len(tokens_all)
            n_syms = min(len(codewords_prefix), n_syms_total)
            codewords_prefix = codewords_prefix[:n_syms]

            bits_intended_rep = sum(len(cw) * reps for cw in codewords_prefix)
            data_bits_to_embed = "".join(cw * reps for cw in codewords_prefix)

            # ---- decode from model (embedding space) ----
            rem = len(data_bits_to_embed) % attacker_n_bits
            padded_embed_bits = data_bits_to_embed + ("0" * (attacker_n_bits - rem) if rem else "")
            padded_bits_written = min(len(padded_embed_bits), (capacity_bits // attacker_n_bits) * attacker_n_bits)

            start_decoding = time.time()
            exfil_padded = decode_all_parameters(defended_sd, attacker_n_bits, padded_bits_written)
            decoding_time = time.time() - start_decoding

            exfil_bits_raw = exfil_padded[:len(data_bits_to_embed)]
            decoded_redundant_bits_used = len(exfil_bits_raw)

            recovered_huff_bits, vote_dbg = _majority_vote_codewords(
                exfil_bits_raw, codewords_prefix, reps=reps
            )

            decoded_syms, leftover = _huffman_decode_bits(recovered_huff_bits, codebook)

            intended_syms = tokens_all[:n_syms]
            payload_bits_intended = "".join(format(t, f"0{w}b") for t in intended_syms)
            payload_bits_decoded  = "".join(format(t, f"0{w}b") for t in decoded_syms)[:len(payload_bits_intended)]

            redundancy_dbg = {
                "reps": int(reps),
                "unit": "huffman_codeword_per_token",
                "symbols_prefix": int(n_syms),
                "decoded_symbols": int(len(decoded_syms)),
                "huffman_leftover_bits": int(leftover),
                **(vote_dbg or {}),
                "decoded_redundant_bits_used": int(decoded_redundant_bits_used),
            }
    # ============================================================
    # APPROACH C: HAMMING(15,11) + interleave, operating on COMPRESSED bits
    # ============================================================
    elif approach == "hamming":
        interleave_cols = int(hamming_interleave_cols)

        coded_full, enc_dbg = hamming15_11_encode_bits(compressed_bits_full)  # noqa: F821
        dataset_bits_total_coded = len(coded_full)

        max_interleaved_len = (capacity_bits // interleave_cols) * interleave_cols
        if max_interleaved_len == 0:
            raise ValueError(f"capacity_bits={capacity_bits} too small for interleave_cols={interleave_cols}")

        coded_seg_len = min(dataset_bits_total_coded, max_interleaved_len)
        coded_seg = coded_full[:coded_seg_len]

        interleaved_seg, pad_interleave = interleave_bits(coded_seg, interleave_cols)  # noqa: F821

        dataset_bits_total = len(interleaved_seg)

        bits_intended_embed = min(dataset_bits_total, capacity_bits)
        interleaved_to_embed = interleaved_seg[:bits_intended_embed]

        rem = len(interleaved_to_embed) % attacker_n_bits
        padded_embed_bits = interleaved_to_embed + ("0" * (attacker_n_bits - rem) if rem else "")

        padded_bits_written = min(len(padded_embed_bits), (capacity_bits // attacker_n_bits) * attacker_n_bits)

        start_decoding = time.time()
        exfil_padded = decode_all_parameters(defended_sd, attacker_n_bits, padded_bits_written)  # noqa: F821
        decoding_time = time.time() - start_decoding

        exfil_interleaved = exfil_padded[:len(interleaved_to_embed)]

        L_blk = (len(exfil_interleaved) // interleave_cols) * interleave_cols
        exfil_interleaved_blk = exfil_interleaved[:L_blk]

        exfil_coded_padded = deinterleave_bits(exfil_interleaved_blk, cols=interleave_cols)  # noqa: F821
        coded_for_decode = exfil_coded_padded[: (len(exfil_coded_padded) // 15) * 15]

        max_plain_bits_decodable = (len(coded_for_decode) // 15) * 11
        plain_target_len = min(dataset_bits_total_compressed, max_plain_bits_decodable)

        recovered_compressed_bits, dec_dbg = hamming15_11_decode_bits(  # noqa: F821
            coded_for_decode,
            plain_bits_len=plain_target_len,
        )
        orig_compressed_bits_cmp = compressed_bits_full[:len(recovered_compressed_bits)]

        bits_intended = min(dataset_bits_total_compressed, (coded_seg_len // 15) * 11)
        dataset_bits_total = dataset_bits_total_compressed

        hamming_dbg = {
            "hamming": {"code_n": 15, "data_k": 11},
            "interleave_cols": int(interleave_cols),
            "interleave_pad_bits": int(pad_interleave),
            "encode": enc_dbg,
            "decode": dec_dbg,
            "coded_seg_len": int(coded_seg_len),
            "interleaved_to_embed_len": int(len(interleaved_to_embed)),
            "interleaved_blk_len": int(L_blk),
            "coded_for_decode_len": int(len(coded_for_decode)),
        }

    # ============================================================
    # APPROACH D: RS fixed-block + interleaving, operating on COMPRESSED bits
    # ============================================================
    elif approach == "rs":
        cfg_rs = dict(rs_cfg or {})
        cfg_rs.setdefault("k_bytes", 16)
        cfg_rs.setdefault("nsym", 32)
        cfg_rs.setdefault("interleave_cols", 32)
        cfg_rs.setdefault("keep_raw_on_fail", True)

        k_bytes = int(cfg_rs["k_bytes"])
        nsym = int(cfg_rs["nsym"])
        interleave_cols = int(cfg_rs["interleave_cols"])
        keep_raw_on_fail = bool(cfg_rs["keep_raw_on_fail"])
        
        payload_is_huffman_bits = bool(huffman_enabled)
        n_bytes = k_bytes + nsym
        bits_per_cw = n_bytes * 8  # RS codeword length (n_bytes)

        # 1) RS encode full compressed dataset
        coded_bits_full, dbg_enc = rs_fixed_block_encode_bits(compressed_bits_full, cfg_rs)
        coded_bits_total_full = len(coded_bits_full)

        # Full dataset size in embedding space (RS-coded + interleaved, block-aligned)
        coded_full_blk_len = (coded_bits_total_full // interleave_cols) * interleave_cols
        coded_full_blk = coded_bits_full[:coded_full_blk_len]
        interleaved_full, pad_interleave_full = interleave_bits(coded_full_blk, interleave_cols)
        dataset_bits_total_rep_full = len(interleaved_full)  # <-- embedding-space dataset total

        # 2) Determine how much we intend to embed (aligned to interleave_cols)
        max_interleaved_len = (capacity_bits // interleave_cols) * interleave_cols
        if max_interleaved_len == 0:
            raise ValueError(f"capacity_bits={capacity_bits} too small for interleave_cols={interleave_cols}")

        coded_seg_len = min(coded_bits_total_full, max_interleaved_len)
        coded_seg = coded_bits_full[:coded_seg_len]

        interleaved_seg, pad_interleave = interleave_bits(coded_seg, interleave_cols)
        bits_intended_embed = min(len(interleaved_seg), capacity_bits)

        # ensure intended embed length is block-aligned (so deinterleave/RS decoding stays consistent)
        bits_intended_embed = (bits_intended_embed // interleave_cols) * interleave_cols

        interleaved_to_embed = interleaved_seg[:bits_intended_embed]
        bits_intended_rep = len(interleaved_to_embed)  # <-- embedding-space intended bits

        # 3) Compute INTENDED payload length in "payload space" (compressed bits)
        # Each RS codeword contributes k_bytes*8 data bits
        n_codewords_intended = coded_seg_len // bits_per_cw
        compressed_bits_intended = min(dataset_bits_total_compressed, n_codewords_intended * (k_bytes * 8))

        # Intended payload bitstream (reference) in payload space
        orig_compressed_bits_cmp = compressed_bits_full[:compressed_bits_intended]

        # 4) Decode from defended model
        rem = len(interleaved_to_embed) % attacker_n_bits
        padded_embed_bits = interleaved_to_embed + ("0" * (attacker_n_bits - rem) if rem else "")
        padded_bits_written = min(len(padded_embed_bits), (capacity_bits // attacker_n_bits) * attacker_n_bits)

        start_decoding = time.time()
        exfil_padded = decode_all_parameters(defended_sd, attacker_n_bits, padded_bits_written)
        decoding_time = time.time() - start_decoding

        exfil_interleaved = exfil_padded[:len(interleaved_to_embed)]
        L_blk = (len(exfil_interleaved) // interleave_cols) * interleave_cols
        exfil_interleaved_blk = exfil_interleaved[:L_blk]

        exfil_coded_padded = deinterleave_bits(exfil_interleaved_blk, cols=interleave_cols)
        coded_for_decode = exfil_coded_padded[: (len(exfil_coded_padded) // bits_per_cw) * bits_per_cw]

        cfg_rs["keep_raw_on_fail"] = keep_raw_on_fail
        recovered_compressed_bits_all, dbg_dec = rs_fixed_block_decode_bits(coded_for_decode, cfg_rs)

        # Decoded payload bitstream in payload space (trim to intended payload length)
        recovered_compressed_bits = recovered_compressed_bits_all[:compressed_bits_intended]

        # ---- Mapping for unified metrics block ----
        payload_bits_intended = orig_compressed_bits_cmp
        payload_bits_decoded  = recovered_compressed_bits

        rs_dbg = {
            "rs": {
                "k_bytes": int(k_bytes),
                "nsym": int(nsym),
                "n_bytes": int(n_bytes),
                "keep_raw_on_fail": bool(keep_raw_on_fail),
            },
            "interleave_cols": int(interleave_cols),
            "interleave_pad_bits": int(pad_interleave),
            "encode": dbg_enc,
            "decode": dbg_dec,
            "coded_bits_total_full": int(coded_bits_total_full),
            "coded_bits_used": int(coded_seg_len),
            "n_codewords_intended": int(n_codewords_intended),
            "compressed_bits_intended": int(compressed_bits_intended),
            "interleaved_to_embed_len": int(len(interleaved_to_embed)),
            "interleaved_blk_len": int(L_blk),
            "coded_for_decode_len": int(len(coded_for_decode)),
            "decoded_compressed_bits_len": int(len(recovered_compressed_bits)),
        }

    else:
        raise ValueError(f"Unknown approach={approach} in _evaluate_as_attacker")
    
    # ============================================================
    # Calculate capacity (encoded/total dataset) and similarity metrics for all approaches:
    # Invariants you must ensure per-branch BEFORE this block:
    #   - embed_dataset_bits_total: int
    #       Total dataset size in *embedding space* for coverage_bits
    #       (e.g., redundant/interleaved/RS-coded full dataset length)
    #   - embed_bits_intended: int
    #       How many bits you INTENDED to embed (in embedding space, before padding)
    #   - payload_bits_intended: str
    #       The INTENDED payload bitstream in the representation used for similarity/parsing
    #       (BASE when Huffman OFF; Huffman-compressed bitstream when Huffman ON)
    #   - payload_bits_decoded: str
    #       The DECODED/RECOVERED payload bitstream in the same representation as payload_bits_intended
    # ============================================================

# ============================================================
# Unified capacity + similarity metrics (COMPARABLE across attacks)
# ============================================================

    attack_capacity = int(capacity_bits)

    # --- embedding-space (diagnostic only) ---
    embed_dataset_bits_total = int(dataset_bits_total_rep_full)   # must be set by each branch
    embed_bits_intended      = int(bits_intended_rep)             # must be set by each branch
    coverage_bits_embed = (embed_bits_intended / embed_dataset_bits_total) if embed_dataset_bits_total else 0.0

    # --- payload-space (MAIN comparable capacity) ---
    payload_bits_total = int(dataset_bits_total_compressed if huffman_enabled else dataset_bits_total_base)
    payload_bits_intended_len = int(len(payload_bits_intended))
    coverage_bits = (payload_bits_intended_len / payload_bits_total) if payload_bits_total else 0.0

    # ---- Similarity in payload space (denominator = intended payload bits) ----
    bit_sim, bit_errs, bit_compared = bit_similarity(payload_bits_decoded, payload_bits_intended)

    # ---- Parse tokens/events from intended & decoded payload ----
    if payload_is_huffman_bits:
        # payload bits are Huffman-coded; decode to symbols (events incl delimiter)
        enc_events_all, _ = huffman_bits_to_events(payload_bits_intended, huff_codec, delim_token)
        dec_events_all, _ = huffman_bits_to_events(payload_bits_decoded,  huff_codec, delim_token)
    else:
        # payload bits are BASE fixed-width
        enc_events_all = bits_to_events(payload_bits_intended, chunk_bits=w)
        dec_events_all = bits_to_events(payload_bits_decoded,  chunk_bits=w)

    enc_seqs = events_to_sequences(enc_events_all, delim_token=delim_token)
    dec_seqs = events_to_sequences(dec_events_all, delim_token=delim_token)

    # ---- Token coverage (computed from INTENDED payload only) ----
    events_encoded = sum(len(s) for s in enc_seqs)
    coverage_tokens = (events_encoded / total_events_in_dataset) if total_events_in_dataset else 0.0

    # ---- Exact similarities (denominator = intended sequences/events) ----
    seq_sim, seq_correct, seq_total = sequence_similarity(enc_seqs, dec_seqs)
    evt_sim, evt_correct, evt_total = event_similarity(enc_seqs, dec_seqs)

    # ---- Soft similarity + membership metrics (pad decoded to intended length) ----
    dec_padded = dec_seqs[:len(enc_seqs)] + ([[]] * max(0, len(enc_seqs) - len(dec_seqs)))

    mean_nhd, _ = mean_normalized_hamming_distance(enc_seqs, dec_padded)
    seq_recovery_soft = 1.0 - mean_nhd

    event_membership = event_membership_rate(enc_seqs, dec_padded)
    seq_membership   = sequence_membership_rate(enc_seqs, dec_padded)

    seq_sim_weighted      = seq_similarity_weighted_by_len(enc_seqs, dec_padded)
    seq_sim_unique        = sequence_similarity_unique(enc_seqs, dec_padded)
    seq_membership_unique = sequence_membership_rate_unique(enc_seqs, dec_padded)
    evt_sim_weighted      = evt_sim

    # ---- 11) Return ----
    res: Dict[str, Any] = {
        "capacity": {
            "attack_capacity": int(attack_capacity),

            # payload-space (comparable)
            "bits_intended": int(payload_bits_intended_len),
            "coverage_bits": float(coverage_bits),
            "coverage_tokens": float(coverage_tokens),

            # diagnostics (not comparable, but useful)
            "embed_bits_total": int(embed_dataset_bits_total),
            "embed_bits_intended": int(embed_bits_intended),
            "coverage_bits_embed": float(coverage_bits_embed),

            "preproc_enabled": bool(huffman_enabled),
            "bits_baseline_full": int(dataset_bits_total_base),
            "bits_preproc_full": int(dataset_bits_total_compressed),

            "compression_ratio": compression_ratio,
            "compression_savings_pct": compression_savings_pct,
            "effective_capacity_gain_x": effective_capacity_gain_x,
            "effective_capacity_gain_pct": effective_capacity_gain_pct,       
            },
        "attack_context": {
            "type": "lsb",
            "approach": approach,
            "attacker_n_bits": int(attacker_n_bits),
            "token_bits": int(w),
            "delim_token": int(delim_token),
            "preprocessing": {"huffman": bool(huffman_enabled)},
            **(redundancy_dbg or {}),
        },
        "similarity": {
            "bit_similarity": float(bit_sim),
            "bit_errors": int(bit_errs),
            "bits_compared": int(bit_compared),
            "bits_decoded": int(len(payload_bits_decoded)),

            "event_similarity": float(evt_sim),
            "events_correct": int(evt_correct),
            "events_compared": int(evt_total),

            "sequence_similarity_exact": float(seq_sim),
            "sequences_correct": int(seq_correct),
            "sequences_compared": int(seq_total),
            "sequence_similarity_soft": float(seq_recovery_soft),

            "event_membership_rate": float(event_membership),
            "sequence_membership_rate": float(seq_membership),
            "event_similarity_weighted": float(evt_sim_weighted),
            "sequence_similarity_weighted": float(seq_sim_weighted),
            "sequence_similarity_unique": float(seq_sim_unique),
            "sequence_membership_rate_unique": float(seq_membership_unique),
        },
        "timing": {
            "decoding_time_s": float(decoding_time),
        },
        "paths": {
            "report_dir": str(out_dir),
        },
    }

    if approach == "redundant":
        res["debug_redundancy"] = {
            **(vote_dbg or {}),
            "decoded_redundant_bits_used": int(decoded_redundant_bits_used or 0),
        }
    if approach == "hamming":
        res["debug_hamming"] = hamming_dbg or {}
    if approach == "rs":
        res["debug_rs"] = rs_dbg or {}

    return res


# ============================================================
# Main entry: run defense
# ============================================================


def run_lsb_defense(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:

    ds = cfg.get("dataset", {})
    if "train_sequences_path" in ds:
        train_source = Path(ds["train_sequences_path"])
    else:
        # fallback for include-style dataset config
        base_dir = Path(ds["base_dir"]) / str(cfg["experiment"]["target_id"])
        train_source = base_dir / ds["train_sequences_file"]

    target_id = str(cfg.get("experiment", {}).get("target_id", "split0"))
    dataset_id = (
    cfg.get("dataset", {}).get("name")
    or cfg.get("experiment", {}).get("dataset")
    or "unknown"
    )

    num_classes = int(options["num_classes"])
    w = event_bit_width(num_classes)  # noqa: F821
    delim_token = (1 << w) - 1

    dp = cfg.get("defense", {}).get("params", {})
    assert dp.get("enabled", True), "Defense not enabled"

    attacked_ckpt = Path(dp["attacked_ckpt"])
    n_bits_defended = int(dp["n_bits_defended"])
    seed = dp.get("seed", None)
    if seed is None:
        seed = cfg.get("experiment", {}).get("seed_defense", None)

    attacker_cfg = dp.get("attacker", {})
    if not attacker_cfg:
        raise KeyError("Missing defense.params.attacker in YAML")

    attacker_n_bits = int(attacker_cfg.get("n_bits", 21))

    rs_cfg = attacker_cfg.get("rs", {}) or {}

    attacker_approach = str(attacker_cfg.get("approach", "baseline"))
    if attacker_approach not in ("baseline", "redundant", "hamming", "rs", "repetition"):
        raise ValueError(f"Unknown attacker.approach: {attacker_approach}")
    
        # preprocessing (shared across attacks)
    pre_att = attacker_cfg.get("preprocessing", {}) or cfg.get("attack", {}).get("preprocessing", {}) or {}
    huffman_enabled = bool(pre_att.get("huffman", False))

    redundant_cfg = attacker_cfg.get("redundant", {}) or {}
    redundancy_block_bits = int(redundant_cfg.get("redundancy_block_bits", 1024))
    token_repeat = int(redundant_cfg.get("reps", 3))

    repetition_cfg = attacker_cfg.get("repetition", {}) or {}
    repetition_reps = int(repetition_cfg.get("reps", token_repeat))


    # used only for hamming; must match attack
    hamming_cfg = attacker_cfg.get("hamming", {}) or {}
    hamming_interleave_cols = int(hamming_cfg.get("interleave_cols", 32))

    debug_k = int(attacker_cfg.get("debug_k", 2))

    assert attacked_ckpt.exists(), f"attacked_ckpt not found: {attacked_ckpt}"
    attacked_sd = _load_state_dict(attacked_ckpt)  # noqa: F821

    # ---------------------------
    # 1) Apply defense overwrite
    # ---------------------------
    start_def = time.time()
    defense_capacity_bits = count_capacity_bits(attacked_sd, n_bits_defended)  # noqa: F821
    rand_stream = _rand_bits(defense_capacity_bits, seed=seed)  # noqa: F821

    remaining = rand_stream
    defended_sd: Dict[str, torch.Tensor] = {}
    for name, param in attacked_sd.items():
        if remaining:
            defended_param, remaining, _ = insert_data_into_tensor_all_bits(
                param, remaining, replace_last_n=n_bits_defended
            )
        else:
            defended_param = param
        defended_sd[name] = defended_param

    defense_time = time.time() - start_def
    rand_bits_written = defense_capacity_bits - len(remaining)

    # ---------------------------
    # 2) Save defended checkpoint
    # ---------------------------

    default_save_dir = f"models/defended/white_box/{dataset_id}/{target_id}"
    base_out = Path(cfg.get("experiment", {}).get("save_dir", default_save_dir))
    _ensure_dir(base_out)

    run_dir = base_out / (
        f"{dataset_id}__def=lsb_rand(n{n_bits_defended})"
        f"__from={attacked_ckpt.parent.name}"
        f"__{time.strftime('%Y%m%d_%H%M%S')}"
    )
    _ensure_dir(run_dir)

    defended_ckpt_path = run_dir / "ckpt_defended.pth"
    torch.save({"state_dict": defended_sd}, defended_ckpt_path)


    # ---------------------------
    # 3) Benign eval
    # ---------------------------
    # Model = deeplog(
    #     input_size=options["input_size"],
    #     hidden_size=options["hidden_size"],
    #     num_layers=options["num_layers"],
    #     num_keys=options["num_classes"],
    # )
#    P, R, F1, FP, FN, prediction_time = evaluate_model(  # noqa: F821
#        Model, options, defended_ckpt_path
#    )

    # ---------------------------
    # 4) Evaluate defense as attacker (baseline vs redundant differ ONLY in encode/decode)
    # ---------------------------
    eval_res = _evaluate_as_attacker(
        defended_sd=defended_sd,
        train_source=train_source,
        attacker_n_bits=attacker_n_bits,
        approach=attacker_approach,
        token_repeat=token_repeat,
        redundancy_block_bits=redundancy_block_bits,   # NEW
        huffman_enabled=huffman_enabled,               # NEW
        repetition_reps=repetition_reps,               # NEW
        hamming_interleave_cols=hamming_interleave_cols,
        rs_cfg=rs_cfg,
        options=options,
        out_dir=run_dir,
        debug_k=debug_k,
    )

    # ---------------------------
    # 5) Final report (keep attack-style keys + defense + utility)
    # ---------------------------
    report: Dict[str, Any] = {
        "capacity": eval_res.get("capacity", {}),
        "attack": {
            "type": "lsb",
            "name": cfg.get("attack", {}).get("name", "lsb_encode"),
            "approach": attacker_approach,
            "n_bits": int(attacker_n_bits),
            "huffman": bool(huffman_enabled),
            "target_id": target_id,
            "train_source": str(train_source),
            "checkpoint_path": str(attacked_ckpt),
            "derived_event_bit_width": int(w),
            "derived_delim_token": int(delim_token),
        },
        "defense": {
            "name": cfg.get("defense", {}).get("name", "lsb_random_overwrite"),
            "n_bits_defended": int(n_bits_defended),
            "seed": seed,
            "defense_capacity_bits": int(defense_capacity_bits),
            "rand_bits_written": int(rand_bits_written),
            "defense_time_s": float(defense_time),
        },
        # "utility": {
        #     "precision": float(P),
        #     "recall": float(R),
        #     "f1": float(F1),
        #     "false_positives": int(FP),
        #     "false_negatives": int(FN),
        # },
        "similarity": eval_res.get("similarity", {}),
        "timing": {
   #         "prediction_time_s": float(prediction_time),
            "defense_time_s": float(defense_time),
            "decoding_time_s": float(eval_res.get("timing", {}).get("decoding_time_s", 0.0)),
        },
        "paths": {
            "attacked_ckpt": str(attacked_ckpt),
            "defended_ckpt": str(defended_ckpt_path),
            "report_dir": str(run_dir),
        },
    }

    # keep optional debug block (only redundant)
    if "debug_redundancy" in eval_res:
        report["debug_redundancy"] = eval_res["debug_redundancy"]

    if "debug_hamming" in eval_res:
        report["debug_hamming"] = eval_res["debug_hamming"]

    if "debug_rs" in eval_res:
        report["debug_rs"] = eval_res["debug_rs"]



    with open(run_dir / "report_defense.json", "w") as f:
        json.dump(report, f, indent=2)

    # ---------------------------
    # 6) Optional W&B logging (flat keys, ATTACK-style + extra membership/unique/weighted)
    # ---------------------------
    try:
        import wandb  # type: ignore

        if wandb.run is not None:
            cap = report.get("capacity", {})
            sim = report.get("similarity", {})

            wandb.log({

                # ---- capacity (payload-space; comparable across attacks) ----
                "capacity/attack_capacity": report["capacity"].get("attack_capacity"),
                "capacity/bits_intended": report["capacity"].get("bits_intended"),
                "capacity/coverage_bits": report["capacity"].get("coverage_bits"),
                "capacity/coverage_tokens": report["capacity"].get("coverage_tokens"),

                # ---- embedding diagnostics ----
                "capacity/embed_bits_total": report["capacity"].get("embed_bits_total"),
                "capacity/embed_bits_intended": report["capacity"].get("embed_bits_intended"),
                "capacity/coverage_bits_embed": report["capacity"].get("coverage_bits_embed"),

                # ---- preprocessing / compression ----
                "capacity/preproc_enabled": int(huffman_enabled),
                "capacity/bits_baseline_full": report["capacity"].get("bits_baseline_full"),
                "capacity/bits_preproc_full": report["capacity"].get("bits_preproc_full"),

                "capacity/compression_ratio": report["capacity"].get("compression_ratio"),
                "capacity/compression_savings_pct": report["capacity"].get("compression_savings_pct"),
                "capacity/effective_capacity_gain_x": report["capacity"].get("effective_capacity_gain_x"),

                "attack/approach": report["attack"]["approach"],
                "attack/n_bits": report["attack"]["n_bits"],
                "attack/target_id": target_id,

                "attack/preproc_huffman": int(huffman_enabled),
                "attack/attacked_ckpt": str(attacked_ckpt),

                "defense/n_bits_defended": report["defense"]["n_bits_defended"],
                "defense/rand_bits_written": report["defense"]["rand_bits_written"],
                "timing/defense_time_s": report["defense"]["defense_time_s"],

                # "utility/precision": report["utility"]["precision"],
                # "utility/recall": report["utility"]["recall"],
                # "utility/f1": report["utility"]["f1"],
                # "utility/fp": report["utility"]["false_positives"],
                # "utility/fn": report["utility"]["false_negatives"],

                "similarity/bit_similarity": sim.get("bit_similarity"),
                "similarity/bit_errors": sim.get("bit_errors"),
                "similarity/bits_compared": sim.get("bits_compared"),
                "similarity/bits_decoded": sim.get("bits_decoded"),

                "similarity/event_similarity": sim.get("event_similarity"),
                "similarity/events_correct": sim.get("events_correct"),
                "similarity/events_compared": sim.get("events_compared"),

                "similarity/sequence_similarity_exact": sim.get("sequence_similarity_exact"),
                "similarity/sequences_correct": sim.get("sequences_correct"),
                "similarity/sequences_compared": sim.get("sequences_compared"),
                "similarity/sequence_similarity_soft": sim.get("sequence_similarity_soft"),

                # keep from old defense eval:
                "similarity/event_membership_rate": sim.get("event_membership_rate"),
                "similarity/sequence_membership_rate": sim.get("sequence_membership_rate"),
                "similarity/event_similarity_weighted": sim.get("event_similarity_weighted"),
                "similarity/sequence_similarity_weighted": sim.get("sequence_similarity_weighted"),
                "similarity/sequence_similarity_unique": sim.get("sequence_similarity_unique"),
                "similarity/sequence_membership_rate_unique": sim.get("sequence_membership_rate_unique"),

            #    "timing/prediction_time_s": report["timing"]["prediction_time_s"],
                "timing/decoding_time_s": report["timing"]["decoding_time_s"],
            })
    except Exception:
        pass

    return report