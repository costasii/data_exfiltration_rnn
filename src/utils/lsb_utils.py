
import torch
import struct
import time
import wandb
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from pathlib import Path
import gc
import os
import sys
from typing import Dict, Any, List, Tuple, Optional, Sequence

from collections import defaultdict

sys.path.append('../../')
from src.training.predict import Predicter

import pandas as pd

import torch.nn as nn

import math


############## NEW CODE #################

# -----------------------
# Bit width + encoding
# -----------------------

def event_bit_width(num_classes: int) -> int:
    # +1 ensures at least one unused symbol (reserve all-ones for delimiter)
    return math.ceil(math.log2(num_classes + 1))

def int_to_wbits(x: int, w: int) -> str:
    if x < 0 or x >= (1 << w):
        raise ValueError(f"value {x} out of range for {w} bits")
    return format(x, f"0{w}b")


# -----------------------
# Capacity + bit similarity
# -----------------------

def count_capacity_bits(state_dict: Dict[str, torch.Tensor], n_bits: int) -> int:
    total_elems = 0
    for _, t in state_dict.items():
        if not torch.is_tensor(t):
            continue
        total_elems += int(t.numel())
    return total_elems * int(n_bits)


# -----------------------
# Bitstream -> tokens -> sequences
# -----------------------

def bits_to_events(bits: str, chunk_bits: int) -> List[int]:
    L = (len(bits) // chunk_bits) * chunk_bits
    return [int(bits[i:i + chunk_bits], 2) for i in range(0, L, chunk_bits)]

def events_to_sequences(events: List[int], delim_token: int) -> List[List[int]]:
    seqs: List[List[int]] = []
    cur: List[int] = []
    for e in events:
        if e == delim_token:
            seqs.append(cur)
            cur = []
        else:
            cur.append(e)
    return seqs  # drop unfinished tail (no delimiter)

from typing import Sequence, Tuple

def pad_decoded_sequences_to_encoded(
    enc_seqs: List[Sequence[int]],
    dec_seqs: List[Sequence[int]],
) -> List[Sequence[int]]:
    """
    Returns a decoded list of the same length as enc_seqs.
    Missing decoded sequences become empty sequences.
    Extra decoded sequences are ignored.
    """
    k = len(enc_seqs)
    dec_cut = list(dec_seqs[:k])
    if len(dec_cut) < k:
        dec_cut.extend([[]] * (k - len(dec_cut)))
    return dec_cut
# -----------------------
# Similarities (attack-style)
# -----------------------

from typing import List, Tuple

def bit_similarity(dec_bits: str, enc_bits: str) -> Tuple[float, int, int]:
    """
    Sim_bit = sum_i 1[dec_i == enc_i] / |B_enc|
    Denominator is the number of ENCODED bits.
    Missing decoded bits count as incorrect.
    """
    total = len(enc_bits)
    if total == 0:
        return 1.0, 0, 0
    L = min(total, len(dec_bits))
    err = sum(1 for i in range(L) if dec_bits[i] != enc_bits[i])
    # missing tail in dec_bits => incorrect bits
    err += (total - L)
    return 1.0 - err / total, err, total


def event_similarity(
    enc_seqs: List[List[int]],
    dec_seqs: List[List[int]],
) -> Tuple[float, int, int]:
    """
    Sim_event = sum_{i,j} 1[dec_{i,j} == enc_{i,j}] / |D_enc|
    Denominator is the number of ENCODED events.
    Missing decoded events count as incorrect.
    """
    enc_events = [e for s in enc_seqs for e in s]
    dec_events = [e for s in dec_seqs for e in s]
    total = len(enc_events)
    if total == 0:
        return 1.0, 0, 0
    L = min(total, len(dec_events))
    correct = sum(1 for i in range(L) if dec_events[i] == enc_events[i])
    return correct / total, correct, total


def sequence_similarity(
    enc_seqs: List[List[int]],
    dec_seqs: List[List[int]],
) -> Tuple[float, int, int]:
    """
    Sim_seq = sum_i 1[dec_i == enc_i] / |S_enc|
    Denominator is the number of ENCODED sequences.
    Missing decoded sequences count as incorrect.
    """
    total = len(enc_seqs)
    if total == 0:
        return 1.0, 0, 0
    correct = 0
    for i, s_enc in enumerate(enc_seqs):
        if i < len(dec_seqs) and dec_seqs[i] == s_enc:
            correct += 1
    return correct / total, correct, total

def normalized_hamming_distance(a: Sequence[int], b: Sequence[int]) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    m = min(la, lb)
    mism = sum(1 for i in range(m) if a[i] != b[i])
    length_penalty = abs(la - lb)
    denom = max(la, lb)
    return (mism + length_penalty) / denom

def mean_normalized_hamming_distance(
    orig_seqs: List[Sequence[int]],
    dec_seqs: List[Sequence[int]],
) -> Tuple[float, List[float]]:
    assert len(orig_seqs) == len(dec_seqs), "orig/dec must be aligned and same count"
    per = [normalized_hamming_distance(o, d) for o, d in zip(orig_seqs, dec_seqs)]
    mean_nhd = sum(per) / len(per) if per else 0.0
    return mean_nhd, per


################### REDUNDANT attack helpers ########################
def repeat_tokens_in_bitstream(bitstream: str, token_bits: int, reps: int) -> str:
    """
    bitstream is a concatenation of token_bits tokens.
    Returns bitstream where each token is repeated reps times.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    if len(bitstream) % token_bits != 0:
        raise ValueError(f"bitstream length {len(bitstream)} not divisible by token_bits={token_bits}")

    out: list[str] = []
    for i in range(0, len(bitstream), token_bits):
        tok = bitstream[i:i + token_bits]
        out.extend([tok] * reps)
    return "".join(out)


def majority_vote_repeated_tokens(bitstream: str, token_bits: int, reps: int) -> Tuple[str, Dict[str, int]]:
    """
    Majority vote per bit across `reps` copies of each token_bits token.
    Input bitstream is in groups of (token_bits * reps).
    Output is base-space bitstream (token_bits per token).
    """
    group_bits = token_bits * reps
    n_groups = len(bitstream) // group_bits

    recovered: list[str] = []
    tokens_with_any_disagreement = 0
    total_bit_disagreements = 0

    for gi in range(n_groups):
        base = gi * group_bits
        copies = [
            bitstream[base + r * token_bits : base + (r + 1) * token_bits]
            for r in range(reps)
        ]

        voted: list[str] = []
        any_disagreement_this_token = False

        for j in range(token_bits):
            ones = sum(1 for r in range(reps) if copies[r][j] == "1")
            maj = "1" if ones >= (reps // 2 + 1) else "0"
            voted.append(maj)

            first = copies[0][j]
            if any(copies[r][j] != first for r in range(1, reps)):
                any_disagreement_this_token = True
                total_bit_disagreements += 1

        if any_disagreement_this_token:
            tokens_with_any_disagreement += 1

        recovered.append("".join(voted))

    dbg = {
        "token_bits": int(token_bits),
        "reps": int(reps),
        "groups_total": int(n_groups),
        "tokens_with_any_disagreement": int(tokens_with_any_disagreement),
        "total_bit_disagreements": int(total_bit_disagreements),
        "dropped_tail_bits": int(len(bitstream) - n_groups * group_bits),
    }
    return "".join(recovered), dbg

############# extra similarity metrics #################
def seq_similarity_weighted_by_len(orig_seqs: List[List[int]], dec_seqs: List[List[int]]) -> float:
    if not orig_seqs or not dec_seqs:
        return 0.0
    m = min(len(orig_seqs), len(dec_seqs))
    num = 0
    den = 0
    for i in range(m):
        w = len(orig_seqs[i])
        den += w
        if orig_seqs[i] == dec_seqs[i]:
            num += w
    return (num / den) if den else 0.0


def event_membership_rate(orig_seqs: List[List[int]], dec_seqs: List[List[int]]) -> float:
    orig_vocab = {e for s in orig_seqs for e in s}
    dec_events = [e for s in dec_seqs for e in s]
    if not dec_events:
        return 0.0
    ok = sum(1 for e in dec_events if e in orig_vocab)
    return ok / len(dec_events)


def sequence_membership_rate(orig_seqs: List[List[int]], dec_seqs: List[List[int]]) -> float:
    orig_set = {tuple(s) for s in orig_seqs}
    if not dec_seqs:
        return 0.0
    ok = sum(1 for s in dec_seqs if tuple(s) in orig_set)
    return ok / len(dec_seqs)


def sequence_similarity_unique(orig_seqs: List[List[int]], dec_seqs: List[List[int]]) -> float:
    orig_set = {tuple(s) for s in orig_seqs}
    dec_set  = {tuple(s) for s in dec_seqs}
    if not orig_set and not dec_set:
        return 1.0
    if not orig_set or not dec_set:
        return 0.0
    return len(orig_set & dec_set) / len(orig_set | dec_set)


def sequence_membership_rate_unique(orig_seqs: List[List[int]], dec_seqs: List[List[int]]) -> float:
    orig_set = {tuple(s) for s in orig_seqs}
    dec_unique = {tuple(s) for s in dec_seqs}
    if not dec_unique:
        return 0.0
    ok = sum(1 for s in dec_unique if s in orig_set)
    return ok / len(dec_unique)


################### defense utils ####################
def _load_state_dict_any(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint must be a state_dict or dict containing key 'state_dict'.")
    return obj

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _rand_bits(n: int, seed: Optional[int] = None) -> str:
    if n <= 0:
        return ""
    if seed is None:
        b = os.urandom((n + 7) // 8)
    else:
        import random
        rng = random.Random(int(seed))
        b = bytes(rng.getrandbits(8) for _ in range((n + 7) // 8))
    bits = "".join(f"{byte:08b}" for byte in b)
    return bits[:n]


################# hamming helpers ########################
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import math
import torch


# ------------------------
# Interleaver (transpose)
# ------------------------

def interleave_bits(bits: str, cols: int) -> Tuple[str, int]:
    """
    Block interleaver via matrix transpose:
      - pad with zeros to fill last row of length cols
      - write out column-wise

    Returns: (interleaved_bits, pad_bits_added)
    """
    if cols <= 1 or not bits:
        return bits, 0

    pad = (-len(bits)) % cols
    if pad:
        bits = bits + ("0" * pad)

    rows = len(bits) // cols
    out = []
    for c in range(cols):
        for r in range(rows):
            out.append(bits[r * cols + c])
    return "".join(out), pad

def deinterleave_bits(bits: str, cols: int) -> str:
    if cols <= 1 or not bits:
        return bits
    if len(bits) % cols != 0:
        raise ValueError(f"deinterleave_bits: len(bits)={len(bits)} not divisible by cols={cols}")
    rows = len(bits) // cols
    out = ["0"] * (rows * cols)
    idx = 0
    for c in range(cols):
        for r in range(rows):
            out[r * cols + c] = bits[idx]
            idx += 1
    return "".join(out)

# ------------------------
# Hamming(15,11) core
# ------------------------
# Positions are 1-indexed:
# parity bits at positions 1,2,4,8
# data bits at positions 3,5,6,7,9,10,11,12,13,14,15

_H15_P = (1, 2, 4, 8)
_H15_D = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)

def _h15_syndrome(code15: str) -> int:
    """
    Returns syndrome in [0..15], using EVEN parity checks.
    syndrome==0 => no single-bit error detected.
    """
    assert len(code15) == 15
    # bits list with 1-indexing convenience
    b = [0] + [1 if ch == "1" else 0 for ch in code15]

    syn = 0
    # parity check for p in {1,2,4,8}:
    # XOR of all positions i where i has that bit set should be 0 (even parity).
    for p in _H15_P:
        s = 0
        for i in range(1, 16):
            if i & p:
                s ^= b[i]
        if s != 0:
            syn |= p
    return syn


def hamming15_11_encode_bits(data_bits: str) -> Tuple[str, Dict[str, Any]]:
    """
    Encode arbitrary bitstring using Hamming(15,11), blockwise.
    Pads the final data block with zeros to 11 bits.

    Returns (coded_bits, debug)
    """
    if not data_bits:
        return "", {"blocks": 0, "data_pad_bits": 0}

    # pad data to multiple of 11
    data_pad = (-len(data_bits)) % 11
    if data_pad:
        data_bits = data_bits + ("0" * data_pad)

    blocks = len(data_bits) // 11
    out = []

    for bi in range(blocks):
        d = data_bits[bi * 11 : (bi + 1) * 11]
        # build codeword positions 1..15
        c = ["0"] * 16  # ignore index 0
        # place data bits
        for j, pos in enumerate(_H15_D):
            c[pos] = d[j]

        # compute parity bits (even parity)
        # For each parity position p, parity is XOR of covered bits excluding itself,
        # then set parity bit so total XOR over covered bits becomes 0.
        for p in _H15_P:
            s = 0
            for i in range(1, 16):
                if i & p and i != p:
                    s ^= (1 if c[i] == "1" else 0)
            c[p] = "1" if s == 1 else "0"

        out.append("".join(c[1:]))

    dbg = {"blocks": int(blocks), "data_pad_bits": int(data_pad)}
    return "".join(out), dbg


def hamming15_11_decode_bits(coded_bits: str, *, plain_bits_len: int) -> Tuple[str, Dict[str, Any]]:
    """
    Decode Hamming(15,11) coded bitstring blockwise.
    Corrects up to 1 bit per 15-bit codeword.
    Drops trailing incomplete 15-bit block.
    Trims output to plain_bits_len.

    Returns (recovered_plain_bits, debug)
    """
    if not coded_bits:
        return "", {
            "blocks_total": 0, "blocks_corrected": 0, "blocks_uncorrectable": 0,
            "corrected_bits_total": 0, "dropped_tail_bits": 0
        }

    blocks_total = len(coded_bits) // 15
    dropped_tail = len(coded_bits) - blocks_total * 15

    blocks_corrected = 0
    blocks_uncorrectable = 0
    corrected_bits_total = 0

    out = []

    for bi in range(blocks_total):
        c15 = coded_bits[bi * 15 : (bi + 1) * 15]
        syn = _h15_syndrome(c15)

        # correct if single-bit error indicated
        if syn != 0 and 1 <= syn <= 15:
            # flip bit at position syn (1-indexed)
            lst = list(c15)
            idx0 = syn - 1
            lst[idx0] = "0" if lst[idx0] == "1" else "1"
            c15_corr = "".join(lst)

            # re-check: if syndrome still nonzero, treat as uncorrectable (>=2 errors)
            syn2 = _h15_syndrome(c15_corr)
            if syn2 == 0:
                blocks_corrected += 1
                corrected_bits_total += 1
                c15 = c15_corr
            else:
                blocks_uncorrectable += 1
                # keep uncorrected c15; you could also skip block, but keep for consistent length

        # extract 11 data bits
        data11 = "".join(c15[pos - 1] for pos in _H15_D)
        out.append(data11)

    recovered = "".join(out)
    recovered = recovered[:plain_bits_len]

    dbg = {
        "blocks_total": int(blocks_total),
        "blocks_corrected": int(blocks_corrected),
        "blocks_uncorrectable": int(blocks_uncorrectable),
        "corrected_bits_total": int(corrected_bits_total),
        "dropped_tail_bits": int(dropped_tail),
        "decoded_plain_bits_len": int(len(recovered)),
    }
    return recovered, dbg

############## ECC RS helpers ############
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

def _require_reedsolo():
    try:
        import reedsolo  # type: ignore
    except Exception as e:
        raise ImportError(
            "reedsolo is required for RS approach. Install with:\n"
            "  pip install reedsolo\n"
            "or\n"
            "  conda install -c conda-forge reedsolo\n"
        ) from e
    return reedsolo

def _bits_to_bytes(bitstr: str) -> bytes:
    if not bitstr:
        return b""
    pad = (-len(bitstr)) % 8
    if pad:
        bitstr = bitstr + ("0" * pad)
    return int(bitstr, 2).to_bytes(len(bitstr) // 8, byteorder="big")

def _bytes_to_bits(b: bytes) -> str:
    return "".join(f"{x:08b}" for x in b)

@dataclass
class RSCfg:
    k_bytes: int = 16
    nsym: int = 8
    interleave_cols: int = 32
    drop_failed_blocks: bool = True
from typing import Any, Dict, Tuple, Union

def rs_fixed_block_encode_bits(plain_bits: str, cfg: Union["RSCfg", Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Encode plain_bits using fixed-size RS blocks:
      - pack bits -> bytes (pad to full bytes)
      - split into k_bytes blocks (pad last with zeros to k_bytes)
      - RS encode each block -> n_bytes = k_bytes + nsym
      - concatenate codewords -> coded_bits
    Returns (coded_bits, debug)
    """
    reedsolo = _require_reedsolo()

    # --- normalize cfg to locals so dict + object both work ---
    if isinstance(cfg, dict):
        k_bytes = int(cfg["k_bytes"])
        nsym = int(cfg["nsym"])
    else:
        k_bytes = int(cfg.k_bytes)
        nsym = int(cfg.nsym)

    assert k_bytes > 0 and nsym > 0
    rsc = reedsolo.RSCodec(nsym)

    plain_bytes = _bits_to_bytes(plain_bits)
    plain_pad_bits = (-len(plain_bits)) % 8  # informational only

    k = k_bytes
    n = k_bytes + nsym

    # pad to full k-byte blocks
    pad_bytes = (-len(plain_bytes)) % k
    if pad_bytes:
        plain_bytes_padded = plain_bytes + (b"\x00" * pad_bytes)
    else:
        plain_bytes_padded = plain_bytes

    blocks = len(plain_bytes_padded) // k
    coded = bytearray()

    for i in range(blocks):
        msg = plain_bytes_padded[i * k:(i + 1) * k]
        cw = rsc.encode(msg)  # msg + ecc
        if len(cw) != n:
            raise RuntimeError(f"RSCodec produced unexpected length {len(cw)} != {n}")
        coded.extend(cw)

    dbg = {
        "k_bytes": int(k_bytes),
        "nsym": int(nsym),
        "n_bytes": int(n),
        "plain_bits_len": int(len(plain_bits)),
        "plain_pad_bits_to_byte": int(plain_pad_bits),
        "plain_bytes_len": int(len(plain_bytes)),
        "plain_bytes_pad_to_k": int(pad_bytes),
        "blocks": int(blocks),
        "coded_bytes_len": int(len(coded)),
        "coded_bits_len": int(len(coded) * 8),
    }
    return _bytes_to_bits(bytes(coded)), dbg

from typing import Any, Dict, Tuple, Union

def rs_fixed_block_decode_bits(coded_bits: str, cfg: Union["RSCfg", Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Decode coded_bits using fixed-size RS blocks (n_bytes = k_bytes+nsym):
      - bits -> bytes (drops trailing incomplete byte)
      - split into n-byte codewords
      - RS decode each codeword independently
      - on failure:
          * if drop_failed_blocks: drop the whole block (do not append)
          * else: append either raw cw[:k] (if keep_raw_on_fail) or zeros
    Returns (decoded_plain_bits, debug)
    """
    reedsolo = _require_reedsolo()

    # --- normalize cfg to locals so dict + object both work ---
    if isinstance(cfg, dict):
        k_bytes = int(cfg["k_bytes"])
        nsym = int(cfg["nsym"])
        drop_failed_blocks = bool(cfg.get("drop_failed_blocks", False))
        keep_raw_on_fail = bool(cfg.get("keep_raw_on_fail", True))
    else:
        k_bytes = int(cfg.k_bytes)
        nsym = int(cfg.nsym)
        drop_failed_blocks = bool(getattr(cfg, "drop_failed_blocks", False))
        keep_raw_on_fail = bool(getattr(cfg, "keep_raw_on_fail", True))

    assert k_bytes > 0 and nsym > 0
    rsc = reedsolo.RSCodec(nsym)

    # drop trailing incomplete byte (shouldn’t happen if you track lengths)
    L_bits = (len(coded_bits) // 8) * 8
    coded_bits = coded_bits[:L_bits]
    coded_bytes = _bits_to_bytes(coded_bits)

    k = k_bytes
    n = k_bytes + nsym
    total_codewords = len(coded_bytes) // n
    used_bytes = total_codewords * n
    coded_bytes = coded_bytes[:used_bytes]

    ok = 0
    fail = 0
    decoded = bytearray()

    for i in range(total_codewords):
        cw = coded_bytes[i * n:(i + 1) * n]
        try:
            # reedsolo decode returns (msg, msg+ecc, errata_pos) in many versions
            res = rsc.decode(cw)
            msg = res[0] if isinstance(res, (tuple, list)) else res
            msg = bytes(msg)
            if len(msg) != k:
                raise RuntimeError(f"RS decoded msg len {len(msg)} != k={k}")
            decoded.extend(msg)
            ok += 1
        except Exception:
            fail += 1
            if drop_failed_blocks:
                # drop entirely: keep alignment shorter, but avoids injecting garbage
                continue
            if keep_raw_on_fail:
                decoded.extend(cw[:k])          # keep raw message bytes
            else:
                decoded.extend(b"\x00" * k)     # zero-fill to preserve length/alignment

    dbg = {
        "k_bytes": int(k_bytes),
        "nsym": int(nsym),
        "n_bytes": int(n),
        "codewords_total": int(total_codewords),
        "codewords_ok": int(ok),
        "codewords_failed": int(fail),
        "drop_failed_blocks": bool(drop_failed_blocks),
        "keep_raw_on_fail": bool(keep_raw_on_fail),
        "decoded_bytes_len": int(len(decoded)),
        "decoded_bits_len": int(len(decoded) * 8),
    }
    return _bytes_to_bits(bytes(decoded)), dbg



################## PREPROCESSING - HUFFMAN COMPRESSION ############################
import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class _HNode:
    sym: Optional[int] = None
    left: Optional["__class__"] = None
    right: Optional["__class__"] = None

class HuffmanCodec:
    def __init__(self, codes: Dict[int, str]):
        self.codes = codes
        self.rev = {v: k for k, v in codes.items()}
        # Build decode trie for streaming decode
        self.trie = _HNode()
        for sym, code in codes.items():
            cur = self.trie
            for b in code:
                if b == "0":
                    if cur.left is None: cur.left = _HNode()
                    cur = cur.left
                else:
                    if cur.right is None: cur.right = _HNode()
                    cur = cur.right
            cur.sym = sym

    @staticmethod
    def from_frequencies(freq: Dict[int, int]) -> "HuffmanCodec":
        # Edge case: only one symbol
        if len(freq) == 1:
            (only_sym,) = freq.keys()
            return HuffmanCodec({only_sym: "0"})

        heap: List[Tuple[int, int, _HNode]] = []
        # tie-breaker for determinism
        uid = 0
        for sym, f in sorted(freq.items(), key=lambda x: x[0]):
            heap.append((f, uid, _HNode(sym=sym)))
            uid += 1
        heapq.heapify(heap)

        while len(heap) > 1:
            f1, u1, n1 = heapq.heappop(heap)
            f2, u2, n2 = heapq.heappop(heap)
            parent = _HNode(left=n1, right=n2)
            heapq.heappush(heap, (f1 + f2, uid, parent))
            uid += 1

        root = heap[0][2]
        codes: Dict[int, str] = {}

        def dfs(node: _HNode, path: str):
            if node.sym is not None:
                codes[node.sym] = path or "0"
                return
            dfs(node.left, path + "0")
            dfs(node.right, path + "1")

        dfs(root, "")
        return HuffmanCodec(codes)

    def encode_tokens(self, toks: List[int]) -> str:
        return "".join(self.codes[t] for t in toks)

    def decode_until_delim(self, bits: str, delim_token: int, start: int = 0) -> Tuple[List[int], int]:
        """Decode tokens until delim_token encountered; returns (tokens_without_delim, next_bit_index)."""
        out: List[int] = []
        cur = self.trie
        i = start
        n = len(bits)
        while i < n:
            cur = cur.left if bits[i] == "0" else cur.right
            i += 1
            if cur is None:
                raise ValueError("Invalid Huffman bitstream (fell off trie)")
            if cur.sym is not None:
                if cur.sym == delim_token:
                    return out, i
                out.append(cur.sym)
                cur = self.trie
        raise ValueError("Truncated Huffman stream (delimiter not found)")
    
    from typing import List, Optional, Tuple

def huffman_bits_to_events(
    bits: str,
    codec: HuffmanCodec,
    delim_token: int,
    *,
    max_seqs: Optional[int] = None,
) -> Tuple[List[int], int]:
    """
    Decode a Huffman-coded bitstream into a flat list of tokens INCLUDING delimiter tokens.
    Returns: (events, bits_consumed)

    - Stops cleanly if the tail is truncated (common due to padding).
    - max_seqs optionally limits how many sequences to decode.
    """
    events: List[int] = []
    i = 0
    n = len(bits)
    seqs = 0

    while i < n:
        if max_seqs is not None and seqs >= max_seqs:
            break

        try:
            toks, i2 = codec.decode_until_delim(bits, delim_token, start=i)
        except ValueError:
            # truncated tail / cannot find delimiter -> stop without crashing
            break

        events.extend(toks)
        events.append(delim_token)  # re-insert delimiter so downstream stays unchanged
        i = i2
        seqs += 1

    return events, i


def huffman_bits_to_sequences(
    bits: str,
    codec: HuffmanCodec,
    delim_token: int,
    *,
    one_indexed: bool = True,
    num_classes: Optional[int] = None,
) -> List[List[int]]:
    """
    Convenience: directly return decoded sequences (no delimiter in output sequences).
    Uses decode_until_delim repeatedly; stops on truncated tail.
    """
    seqs: List[List[int]] = []
    i = 0
    n = len(bits)

    while i < n:
        try:
            toks, i = codec.decode_until_delim(bits, delim_token, start=i)
        except ValueError:
            break

        if num_classes is not None:
            for t in toks:
                if t < 0 or t >= num_classes:
                    raise ValueError(f"Decoded token {t} out of range [0, {num_classes-1}]")

        if one_indexed:
            seqs.append([t + 1 for t in toks])
        else:
            seqs.append(toks)

    return seqs

################### redundant helpers ########################


from typing import Dict, Any, Tuple

def repeat_bit_blocks(bits: str, *, block_bits: int, reps: int) -> Tuple[str, Dict[str, Any]]:
    """
    Repeat fixed-size bit blocks reps times: [B1 B2 ...] -> [B1..B1][B2..B2]...
    Pads the last block with zeros to full block_bits (padding is NOT encoded in-stream;
    caller must track intended length separately, as you already do with bits_intended).
    """
    assert reps >= 2
    assert block_bits > 0

    L = len(bits)
    pad = (-L) % block_bits
    if pad:
        bits_padded = bits + ("0" * pad)
    else:
        bits_padded = bits

    n_blocks = len(bits_padded) // block_bits
    out = []
    for i in range(n_blocks):
        blk = bits_padded[i * block_bits:(i + 1) * block_bits]
        out.append(blk * reps)

    dbg = {"block_bits": block_bits, "reps": reps, "pad_bits": pad, "blocks": n_blocks}
    return "".join(out), dbg


def majority_vote_bit_blocks(bits_redundant: str, *, block_bits: int, reps: int) -> Tuple[str, Dict[str, Any]]:
    """
    Inverse of repeat_bit_blocks via bit-wise majority voting within each repeated block.
    Expects layout: [B1..B1][B2..B2]...
    Returns recovered padded bits (caller slices back to intended length).
    """
    assert reps >= 2
    assert block_bits > 0
    group = block_bits * reps
    n_groups = len(bits_redundant) // group

    out = []
    voted = 0
    for g in range(n_groups):
        base = g * group
        # vote each bit position within the block
        blk_bits = []
        for j in range(block_bits):
            ones = 0
            for r in range(reps):
                if bits_redundant[base + r * block_bits + j] == "1":
                    ones += 1
            blk_bits.append("1" if ones > (reps // 2) else "0")
        out.append("".join(blk_bits))
        voted += 1

    dbg = {"block_bits": block_bits, "reps": reps, "groups_voted": voted}
    return "".join(out), dbg

def base_bits_intended_from_compressed_budget(
    base_chunks: List[str],
    comp_chunks: List[str],
    comp_bits_budget: int,
) -> int:
    base_sum = 0
    comp_sum = 0
    for b, c in zip(base_chunks, comp_chunks):
        if comp_sum + len(c) > comp_bits_budget:
            break
        comp_sum += len(c)
        base_sum += len(b)
    return base_sum


############# REPETITION HUFFMAN OFF #################

def _repeat_fixed_wbit_tokens_from_bits(bits: str, *, w: int, reps: int) -> str:
    """Repeat each w-bit token reps times. bits length must be multiple of w."""
    assert reps >= 1
    L = (len(bits) // w) * w
    bits = bits[:L]
    out = []
    for i in range(0, L, w):
        tok = bits[i:i+w]
        out.append(tok * reps)
    return "".join(out)


def _majority_vote_fixed_tokens(repeated_bits: str, *, token_bits: int, reps: int) -> Tuple[str, Dict[str, Any]]:
    """
    Input: repeated stream where each original token (token_bits) was repeated reps times contiguously.
    Output: recovered token stream (one token_bits per original token), using per-bit majority vote.
    """
    assert reps >= 1
    group_bits = token_bits * reps
    n_groups = len(repeated_bits) // group_bits

    recovered = []
    tokens_with_any_disagreement = 0
    total_bit_disagreements = 0

    for g in range(n_groups):
        block = repeated_bits[g*group_bits:(g+1)*group_bits]
        # split into reps copies
        copies = [block[r*token_bits:(r+1)*token_bits] for r in range(reps)]
        # vote bitwise
        voted = []
        any_disagree = False
        for b in range(token_bits):
            ones = sum(1 for r in range(reps) if copies[r][b] == "1")
            zeros = reps - ones
            bit = "1" if ones > zeros else "0"  # ties -> 0
            voted.append(bit)
            if ones not in (0, reps):
                any_disagree = True
                total_bit_disagreements += 1
        if any_disagree:
            tokens_with_any_disagreement += 1
        recovered.append("".join(voted))

    dbg = {
        "groups_decoded": int(n_groups),
        "tokens_with_any_disagreement": int(tokens_with_any_disagreement),
        "total_bit_disagreements": int(total_bit_disagreements),
        "token_bits": int(token_bits),
        "reps": int(reps),
    }
    return "".join(recovered), dbg

#--------- REPETITION HUFFMAN ON ##############
import heapq
from collections import Counter

class _HuffNode:
    __slots__ = ("sym", "freq", "left", "right")
    def __init__(self, sym, freq, left=None, right=None):
        self.sym = sym
        self.freq = freq
        self.left = left
        self.right = right

def _build_huffman_codebook(symbols: List[int]) -> Dict[int, str]:
    """
    Build Huffman codes over integer symbols.
    Returns {symbol: codeword_bits}.
    """
    cnt = Counter(symbols)
    heap = []
    uid = 0
    for sym, f in cnt.items():
        heap.append((f, uid, _HuffNode(sym, f)))
        uid += 1
    heapq.heapify(heap)

    # edge case: only one symbol -> give it "0"
    if len(heap) == 1:
        (_, _, node) = heap[0]
        return {node.sym: "0"}

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        parent = _HuffNode(None, f1 + f2, left=n1, right=n2)
        heapq.heappush(heap, (parent.freq, uid, parent))
        uid += 1

    root = heap[0][2]
    codebook: Dict[int, str] = {}

    def dfs(node: _HuffNode, prefix: str):
        if node.sym is not None:
            codebook[node.sym] = prefix or "0"
            return
        dfs(node.left,  prefix + "0")
        dfs(node.right, prefix + "1")

    dfs(root, "")
    return codebook


def _huffman_encode_symbols(symbols: List[int], codebook: Dict[int, str]) -> Tuple[str, List[str]]:
    """Return concatenated bitstream and per-symbol codeword list."""
    codewords = [codebook[s] for s in symbols]
    return "".join(codewords), codewords


def _repeat_codewords(codewords: List[str], *, reps: int) -> str:
    """Repeat each codeword reps times contiguously."""
    assert reps >= 1
    return "".join(cw * reps for cw in codewords)


def _prefix_by_budget_codewords(codewords: List[str], *, reps: int, bit_budget: int) -> Tuple[List[str], int]:
    """
    Choose the largest prefix of codewords such that sum(len(cw)*reps) <= bit_budget.
    Returns (prefix_codewords, used_bits_in_repeated_space).
    """
    used = 0
    out = []
    for cw in codewords:
        add = len(cw) * reps
        if used + add > bit_budget:
            break
        out.append(cw)
        used += add
    return out, used


def _majority_vote_codewords(repeated_bits: str, codewords_prefix: List[str], *, reps: int) -> Tuple[str, Dict[str, Any]]:
    """
    Majority-vote for each prefix codeword (variable length):
    - assumes repeated_bits is exactly concat(cw*reps for cw in codewords_prefix)
    - votes per bit position across the reps copies of that cw
    Returns recovered concatenated Huffman bits (one cw per symbol).
    """
    assert reps >= 1
    idx = 0
    recovered = []
    cws_with_any_disagreement = 0
    total_bit_disagreements = 0

    for cw in codewords_prefix:
        L = len(cw)
        group_len = L * reps
        block = repeated_bits[idx: idx + group_len]
        idx += group_len

        copies = [block[r*L:(r+1)*L] for r in range(reps)]
        voted = []
        any_disagree = False
        for b in range(L):
            ones = sum(1 for r in range(reps) if copies[r][b] == "1")
            zeros = reps - ones
            bit = "1" if ones > zeros else "0"
            voted.append(bit)
            if ones not in (0, reps):
                any_disagree = True
                total_bit_disagreements += 1
        if any_disagree:
            cws_with_any_disagreement += 1
        recovered.append("".join(voted))

    dbg = {
        "symbols_prefix": int(len(codewords_prefix)),
        "codewords_with_any_disagreement": int(cws_with_any_disagreement),
        "total_bit_disagreements": int(total_bit_disagreements),
        "reps": int(reps),
    }
    return "".join(recovered), dbg


def _huffman_decode_bits(bitstream: str, codebook: Dict[int, str]) -> Tuple[List[int], int]:
    """
    Decode concatenated Huffman bits back to symbols using the codebook.
    Returns (decoded_symbols, leftover_bits_count).
    """
    inv = {v: k for k, v in codebook.items()}
    decoded: List[int] = []
    cur = ""
    for ch in bitstream:
        cur += ch
        if cur in inv:
            decoded.append(inv[cur])
            cur = ""
    leftover = len(cur)
    return decoded, leftover

# ----------------------------
# Non-Huffman repetition (your old logic)
# ----------------------------
def _repeat_fixed_wbit_tokens(binary_chunks: List[str], w: int, reps: int) -> str:
    """
    Input: list of per-sequence bitstrings; each is concatenation of w-bit tokens
           and ends with a w-bit delimiter token (all-ones).
    Output: concatenated bitstream where each w-bit token is repeated `reps` times.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    out: List[str] = []
    for seq_bits in binary_chunks:
        if len(seq_bits) % w != 0:
            raise ValueError(f"Sequence bits length {len(seq_bits)} not divisible by w={w}")
        for i in range(0, len(seq_bits), w):
            tok = seq_bits[i : i + w]
            if len(tok) != w or any(c not in "01" for c in tok):
                raise ValueError(f"Expected {w}-bit token, got: {tok!r}")
            out.extend([tok] * reps)
    return "".join(out)


def _majority_vote_fixed_tokens(bitstream: str, token_bits: int, reps: int) -> Tuple[str, Dict[str, Any]]:
    """
    Majority vote per bit across `reps` repeated fixed-width tokens (token_bits).
    Returns recovered bitstream in base token space (multiple of token_bits).
    """
    group_bits = token_bits * reps
    n_groups = len(bitstream) // group_bits

    recovered: List[str] = []
    tokens_with_any_disagreement = 0
    total_bit_disagreements = 0

    for gi in range(n_groups):
        base = gi * group_bits
        copies = [
            bitstream[base + r * token_bits : base + (r + 1) * token_bits]
            for r in range(reps)
        ]
        voted: List[str] = []
        any_disagree = False
        for j in range(token_bits):
            ones = sum(1 for r in range(reps) if copies[r][j] == "1")
            maj = "1" if ones >= (reps // 2 + 1) else "0"
            voted.append(maj)
            first = copies[0][j]
            if any(copies[r][j] != first for r in range(1, reps)):
                any_disagree = True
                total_bit_disagreements += 1
        if any_disagree:
            tokens_with_any_disagreement += 1
        recovered.append("".join(voted))

    dbg = {
        "mode": "fixed_wbit",
        "token_bits": int(token_bits),
        "reps": int(reps),
        "groups_total": int(n_groups),
        "tokens_with_any_disagreement": int(tokens_with_any_disagreement),
        "total_bit_disagreements": int(total_bit_disagreements),
        "dropped_tail_bits": int(len(bitstream) - n_groups * group_bits),
    }
    return "".join(recovered), dbg





################## OLD CODE - SOME FUNCTIONS ARE USED, SOME NOT, GO THROUGH AND DELETE: ####################
""" Convert the training data in its binary representation:
    - each event is decreased by 1 to keep consistency
    - each event is represented with 6-bit binary representation
    - each sequence is terminated by the delimiter "111111" 
    - return a list of binary strings
"""

def int_to_6bit(num):
    """Convert an integer (0-33) into a 6-bit binary string."""
    return format(num, '06b')


# def convert_sequences_to_binary(original_sequences):
#     """
#     Convert sequences of integers (already loaded) into binary strings.
#     Each event is mapped to a 6-bit code, and '111111' is appended as an end-of-sequence marker.
    
#     Args:
#         original_sequences (List[List[int]]): List of sequences, each sequence is a list of event IDs.
    
#     Returns:
#         List[str]: List of binary strings, one per sequence.
#     """
#     binary_sequences = []
#     for seq in original_sequences:
#         # subtract 1 from each event (your encoding expects 0-based)
#         seq = [n - 1 for n in seq]

#         binary_sequence = "".join(int_to_6bit(num) for num in seq)
#         binary_sequence += "111111"  # end-of-sequence marker
#         binary_sequences.append(binary_sequence)

#     return binary_sequences

# generic encoding to number of classes

def event_bit_width(num_classes: int) -> int:
    # +1 ensures at least one unused symbol (reserve all-ones for delimiter)
    return math.ceil(math.log2(num_classes + 1))

def int_to_wbits(x: int, w: int) -> str:
    if x < 0 or x >= (1 << w):
        raise ValueError(f"value {x} out of range for {w} bits")
    return format(x, f"0{w}b")

from typing import Any, Dict, List, Optional, Tuple

def convert_sequences_to_binary_huffman(
    original_sequences: List[List[int]],
    num_classes: int,
    *,
    one_indexed: bool = True,
    codec: Optional[HuffmanCodec] = None,
) -> Tuple[List[str], HuffmanCodec]:
    """
    Huffman version that still returns List[str] (one bitstring per sequence),
    with delimiter token appended (same semantics as baseline), plus the codec.
    """
    w = event_bit_width(num_classes)
    delim_token = (1 << w) - 1

    # build token sequences including delimiter
    token_seqs: List[List[int]] = []
    freq: Dict[int, int] = {}
    for seq in original_sequences:
        seq0 = [n - 1 for n in seq] if one_indexed else list(seq)
        for t in seq0:
            if t < 0 or t >= num_classes:
                raise ValueError(f"Event id {t} out of range [0, {num_classes-1}]")
        toks = seq0 + [delim_token]
        token_seqs.append(toks)
        if codec is None:
            for t in toks:
                freq[t] = freq.get(t, 0) + 1

    if codec is None:
        codec = HuffmanCodec.from_frequencies(freq)

    out_bits = [codec.encode_tokens(toks) for toks in token_seqs]
    return out_bits, codec


def convert_sequences_to_binary(
    original_sequences: List[List[int]],
    num_classes: int,
    *,
    one_indexed: bool = True,
) -> List[str]:
    """
    Convert sequences of integers into binary strings.

    - Uses w = event_bit_width(num_classes) = ceil(log2(num_classes + 1))
      so there is always at least one unused symbol.
    - Reserves the all-ones token (2^w - 1) as the delimiter.
    - By default expects events are 1-indexed (DeepLog style), converts to 0-based.
      Set one_indexed=False if your sequences are already 0-based.

    Returns:
        List[str]: one binary string per sequence (events + delimiter at end)
    """
    w = event_bit_width(num_classes)
    delim_token = (1 << w) - 1
    delim_bits = int_to_wbits(delim_token, w)

    out: List[str] = []
    for seq in original_sequences:
        if one_indexed:
            seq0 = [n - 1 for n in seq]
        else:
            seq0 = list(seq)

        # Optional safety: ensure symbols don't collide with delimiter / range
        for n in seq0:
            if n < 0 or n >= num_classes:
                raise ValueError(f"Event id {n} out of range [0, {num_classes-1}]")

        bits = "".join(int_to_wbits(n, w) for n in seq0) + delim_bits
        out.append(bits)

    return out





"""
NAIVE ENCODING
"""

def float_to_bin32_le(f):
    """Convert float to 32-bit binary string (little endian)"""
    b = struct.pack('<f', f)          # float to bytes (little endian)
    i = struct.unpack('<I', b)[0]     # bytes to unsigned int
    return format(i, '032b')          # int to 32-bit binary

def bin32_to_float_le(bstr):
    """Convert 32-bit binary string to float (little endian)"""
    i = int(bstr, 2)
    b = struct.pack('<I', i)
    return struct.unpack('<f', b)[0]

def insert_data_into_tensor_all_bits(tensor, data_bits, replace_last_n):
    """
    Replaces the last `replace_last_n` bits of the *entire 32-bit float representation*
    with bits from `data_bits`. The last float may be partially encoded if not enough bits remain.

    Returns:
        - modified tensor,
        - remaining bits (should be empty if all bits are encoded),
        - count of modified floats.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(tensor)}")

    flat_tensor = tensor.flatten()
    modified_tensor = []
    data_ptr = 0
    encoded_count = 0

    for val in flat_tensor:
        bin_val = float_to_bin32_le(val)

        bits_left = len(data_bits) - data_ptr
        if bits_left == 0:
            # No bits left to encode
            modified_tensor.append(val)
            continue

        # Use only the remaining bits (could be < replace_last_n)
        n_bits = min(replace_last_n, bits_left)
        new_bin_val = bin_val[:-replace_last_n] + data_bits[data_ptr:data_ptr + n_bits].ljust(replace_last_n, '0')
        new_float = bin32_to_float_le(new_bin_val)

        modified_tensor.append(new_float)
        data_ptr += n_bits
        encoded_count += 1

        if data_ptr >= len(data_bits):
            break

    reshaped_tensor = torch.tensor(modified_tensor + list(flat_tensor[len(modified_tensor):]), dtype=tensor.dtype).reshape(tensor.shape)
    remaining_bits = data_bits[data_ptr:]
    return reshaped_tensor, remaining_bits, encoded_count

""" DECODING """

def decode_all_parameters(model_weights, bit_len, bit_target_len):
    """
    Extract exactly `bit_target_len` bits from model parameters, using `bit_len` bits per float.
    Ignores any padding bits that were added to fill the last float during encoding.
    
    Args:
        model_weights: dict of model parameters (e.g., state_dict).
        bit_len: number of bits embedded per float.
        bit_target_len: exact number of bits that were originally encoded.

    Returns:
        A string of exactly `bit_target_len` decoded bits.
    """
    all_bits = ''
    for param in model_weights.values():
        flat = param.cpu().numpy().flatten()
        for val in flat:
            if len(all_bits) >= bit_target_len:
                return all_bits  # stop once enough bits are collected

            bin_val = float_to_bin32_le(val)
            encoded_zone = bin_val[-bit_len:]  # get the bits used for encoding
            remaining_bits = bit_target_len - len(all_bits)
            bits_to_add = encoded_zone[:remaining_bits]  # handle partial last float
            all_bits += bits_to_add
    return all_bits


def bits_to_sequences(bitstream, num_sequences, last_seq_len, delimiter='111111'):
    sequences = []
    current = ''
    last_seq_len = last_seq_len + 1
    for i in range(0, len(bitstream), 6):
        chunk = bitstream[i:i+6]
        if len(chunk) < 6:
            break
        if chunk == delimiter:
            if current:
                # Convert all full 6-bit chunks in current
                seq = [int(current[j:j+6], 2) for j in range(0, len(current), 6)]
                sequences.append(seq)
                current = ''
                if len(sequences) >= num_sequences:
                    break
        else:
            current += chunk

    # Flush leftover bits if no final delimiter
    if current and len(sequences) < num_sequences:
        flushable = current[:len(current) - len(current) % 6]
        if flushable:
            seq = [int(flushable[j:j+6], 2) for j in range(0, len(flushable), 6)]
            sequences.append(seq)

    # Trim last sequence to expected length
    if len(sequences) == num_sequences:
        sequences[-1] = sequences[-1][:last_seq_len]

    # Convert to 1-based indexing
    sequences_final = [[num + 1 for num in sublist] for sublist in sequences]
    return sequences_final

def format_bitstream_by_delimiter(bitstream, delimiter='111111'):
    formatted = ''
    i = 0
    while i <= len(bitstream) - 6:
        chunk = bitstream[i:i+6]
        formatted += chunk
        i += 6
        if chunk == delimiter:
            formatted += '\n'  # insert newline after delimiter
    return formatted

def load_training_sequences(path):
    with open(path, "r") as f:
        return [[int(x) for x in line.strip().split()] for line in f.readlines()]  # apply -1 to match encoding

def verify_decoding_all_metrics(decoded_seqs, original_seqs, save_path_prefix="decoded_output", known_events=33, subseq_len=3):
    matched_sequences = 0
    matched_events = 0
    total_events = sum(len(seq) for seq in original_seqs)
    total_sequences = len(original_seqs)

    exact_event_matches = 0
    sequence_membership_count = 0
    decoded_event_set = set()
    original_event_set = set()
    subsequence_match_count = 0
    total_possible_subsequences = 0

    # For sequence membership (regardless of order)
    decoded_seq_sets = [tuple(seq) for seq in decoded_seqs]
    original_seq_sets = [tuple(seq) for seq in original_seqs]

    for orig_seq in original_seqs:
        original_event_set.update(orig_seq)
        if tuple(orig_seq) in decoded_seq_sets:
            sequence_membership_count += 1

    for dec_seq, orig_seq in zip(decoded_seqs, original_seqs):
        if dec_seq == orig_seq:
            matched_sequences += 1
            matched_events += len(orig_seq)

        event_matches = sum(d == o for d, o in zip(dec_seq, orig_seq))
        exact_event_matches += event_matches

        decoded_event_set.update(dec_seq)

        if subseq_len is not None and len(orig_seq) >= subseq_len:
            total_possible_subsequences += len(orig_seq) - subseq_len + 1
            for i in range(len(orig_seq) - subseq_len + 1):
                orig_sub = orig_seq[i:i+subseq_len]
                dec_sub = dec_seq[i:i+subseq_len]
                if orig_sub == dec_sub:
                    subsequence_match_count += 1

    event_membership_count = len(decoded_event_set & original_event_set)
    event_membership_total = min(known_events, len(original_event_set))

    print(f"\n🧠 Decoding coverage:")
    print(f"   ✅ Exact sequence matches      : {matched_sequences}/{total_sequences} ({matched_sequences / total_sequences:.2%})")
    print(f"   ✅ Exact event position matches: {exact_event_matches}/{total_events} ({exact_event_matches / total_events:.2%})")

    print(f"\n📋 Additional metrics:")
    print(f"   🧩 Sequence membership         : {sequence_membership_count}/{total_sequences} ({sequence_membership_count / total_sequences:.2%})")
    print(f"   🎯 Event membership            : {event_membership_count}/{event_membership_total} ({event_membership_count / event_membership_total:.2%})")

    if subseq_len is not None:
        rate = subsequence_match_count / total_possible_subsequences if total_possible_subsequences > 0 else 0
        print(f"   🔹 Subsequence matches (len={subseq_len}) : {subsequence_match_count}")
        print(f"   🔹 Subsequence match rate     : {subsequence_match_count}/{total_possible_subsequences} ({rate:.2%})")

    # Save decoded and original files
    orig_file = f"{save_path_prefix}_original.txt"
    dec_file = f"{save_path_prefix}_decoded.txt"

    with open(orig_file, "w") as f:
        for line in original_seqs:
            f.write(' '.join(str(bit) for bit in line) + "\n")

    with open(dec_file, "w") as f:
        for line in decoded_seqs:
            f.write(' '.join(str(bit) for bit in line) + "\n")

    print(f"\n📝 Files saved:")
    print(f"   📄 Original sequences → {orig_file}")
    print(f"   📄 Decoded sequences  → {dec_file}")

    return {
        "matched_sequences": matched_sequences,
        "exact_event_matches": exact_event_matches,
        "sequence_membership": sequence_membership_count,
        "event_membership": event_membership_count,
        "subsequence_matches": subsequence_match_count,
        "subsequence_match_total": total_possible_subsequences,
        "subsequence_match_rate": rate,
        "total_sequences": total_sequences,
        "total_events": total_events,
        "event_membership_total": event_membership_total
    }






import torch
import random

def insert_random_bits_into_tensor(tensor, replace_last_n, seed=42):
    """
    Embed random bits into the least significant `replace_last_n` bits of each float's mantissa.
    
    If `seed` is provided, it will be used to make the random bitstream reproducible.
    
    Returns the modified tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(tensor)}")

    if seed is not None:
        random.seed(seed)

    flat_tensor = tensor.flatten()
    modified_tensor = []
    sanitized_count = 0


    for val in flat_tensor:
        bin_val = float_to_bin32_le(val)

        # Generate reproducible random bits
        random_bits = ''.join(random.choice('01') for _ in range(replace_last_n))

        # Replace least significant bits of mantissa
        new_bin_val = bin_val[:-replace_last_n] + random_bits

        new_float = bin32_to_float_le(new_bin_val)

        modified_tensor.append(new_float)

        sanitized_count += 1

    reshaped_tensor = torch.tensor(modified_tensor, dtype=tensor.dtype).reshape(tensor.shape)
    return reshaped_tensor, sanitized_count

def insert_zero_bits_into_tensor(tensor, replace_last_n):
    """
    Embed all-zero bits into the least significant `replace_last_n` bits of each float's 32-bit representation.

    Returns:
        - The modified tensor with zero bits inserted
        - The number of floats modified
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(tensor)}")

    flat_tensor = tensor.flatten()
    modified_tensor = []
    sanitized_count = 0

    for val in flat_tensor:
        bin_val = float_to_bin32_le(val)

        # Generate zero bits
        zero_bits = '0' * replace_last_n

        # Replace least significant bits
        new_bin_val = bin_val[:-replace_last_n] + zero_bits

        new_float = bin32_to_float_le(new_bin_val)
        modified_tensor.append(new_float)
        sanitized_count += 1

    reshaped_tensor = torch.tensor(modified_tensor, dtype=tensor.dtype).reshape(tensor.shape)
    return reshaped_tensor, sanitized_count


def bitwise_levenshtein_distance_from_files(file1, file2):
    """
    Computes the normalized Levenshtein similarity between two bitstreams read from files,
    using a memory-efficient dynamic programming approach (O(n) space).

    Returns:
        float: similarity score in [0, 1]
    """
    # Read and clean bitstreams
    with open(file1, 'r') as f1:
        s = f1.read().replace('\n', '').strip()
    with open(file2, 'r') as f2:
        t = f2.read().replace('\n', '').strip()

    len_s, len_t = len(s), len(t)

    # Always make `s` the shorter string to use less memory
    if len_s > len_t:
        s, t = t, s
        len_s, len_t = len_t, len_s

    # Initialize previous and current row
    previous_row = list(range(len_s + 1))
    current_row = [0] * (len_s + 1)

    for j in range(1, len_t + 1):
        current_row[0] = j
        for i in range(1, len_s + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            current_row[i] = min(
                current_row[i - 1] + 1,     # insertion
                previous_row[i] + 1,        # deletion
                previous_row[i - 1] + cost  # substitution
            )
        previous_row, current_row = current_row, previous_row  # Swap rows

    distance = previous_row[len_s]
    similarity = 1 - (distance / max(len_s, len_t))
    return similarity

def hamming_distance_from_files(file1, file2):
    """
    Computes the normalized Hamming similarity between two bitstreams read from files.
    If bitstreams differ in length, trims the longer one to match the shorter.

    Returns:
        similarity: float in [0, 1]
        distance: int (number of differing bits)
        length: int (number of bits compared)
    """
    # Load and clean bitstreams
    with open(file1, 'r') as f1:
        bits1 = f1.read().replace('\n', '').strip()
    with open(file2, 'r') as f2:
        bits2 = f2.read().replace('\n', '').strip()

    # Match lengths by trimming the longer stream
    min_len = min(len(bits1), len(bits2))
    bits1 = bits1[:min_len]
    bits2 = bits2[:min_len]

    # Compute Hamming distance
    distance = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
    similarity = 1 - (distance / min_len)

    return similarity, distance, min_len


"""
SMART ENCODING
"""

# ========================
# SMART PIPELINE HELPERS
# ========================

from collections import Counter


from collections import Counter

def deduplicate_and_sort_sequences(sequences):
    """
    Deduplicate sequences, record frequencies, and sort by counts (descending).
    
    Args:
        sequences (List[List[int]]): full training set (list of event sequences).
    Returns:
        sorted_sequences (List[List[int]])
        sorted_counts (List[int])
    """
    seq_tuples = [tuple(seq) for seq in sequences]
    counter = Counter(seq_tuples)
    
    # sort by counts, highest first
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    sorted_sequences = [list(seq) for seq, _ in sorted_items]
    sorted_counts = [count for _, count in sorted_items]
    
    total = len(sequences)
    unique = len(sorted_sequences)
    reduction = total - unique
    reduction_pct = 100 * (reduction / total)
    
    print(f"[deduplicate_and_sort_sequences] Total sequences:  {total}")
    print(f"[deduplicate_and_sort_sequences] Unique sequences: {unique}")
    print(f"[deduplicate_and_sort_sequences] Reduction:        {reduction} ({reduction_pct:.2f}%)")
    
    return sorted_sequences, sorted_counts

from typing import List, Tuple, Dict

def serialize_sequences(unique_sequences, counts,
                        num_classes: int) -> Tuple[str, int, int, Dict[str,int]]:
    """
    Encode sequences → bitstream.

    Encoding choices:
      - Each event encoded in fixed w bits (w = event_bit_width(num_classes)).
      - Delimiter = all-ones of length w (reserved, cannot represent an event).
      - Count encoded in fixed width (count_width), which is rounded up to a multiple of w.
      - NEW: delimiter is appended both AFTER the sequence and AFTER the count.
             (sequence_bits) [delim] (count_bits) [delim]

    Returns:
      - bitstream: str of '0'/'1'
      - count_width: int (bits used per count)
      - w: int (bits per event)
      - stats: dict with keys:
          'bits_events', 'bits_delimiters', 'bits_counts', 'total_bits', 
          'n_sequences', 'n_events_total'
    """
    if len(unique_sequences) != len(counts):
        raise ValueError("unique_sequences and counts must have the same length")
    if len(unique_sequences) == 0:
        # nothing to encode
        return "", 0, 0, {
            "bits_events": 0,
            "bits_delimiters": 0,
            "bits_counts": 0,
            "total_bits": 0,
            "n_sequences": 0,
            "n_events_total": 0
        }

    w = event_bit_width(num_classes)            # bits per event
    delim = "1" * w

    max_count = max(counts)
    # round up count bit length to nearest multiple of w
    count_width = ((max_count.bit_length() + w - 1) // w) * w

    parts = []
    n_events_total = 0
    for seq, c in zip(unique_sequences, counts):
        # 1-based -> 0-based
        seq0 = [e - 1 for e in seq]
        # events
        for e in seq0:
            if e >= num_classes or e < 0:
                raise ValueError(f"event {e+1} out of range for num_classes={num_classes}")
            parts.append(int_to_wbits(e, w))   # should return a w-length string
            n_events_total += 1

        # delimiter after sequence
        parts.append(delim)

        # count (fixed-width)
        parts.append(format(c, f"0{count_width}b"))

        # delimiter after count (NEW)
        parts.append(delim)

    bitstream = "".join(parts)

    # Stats
    n_sequences = len(unique_sequences)
    bits_events = n_events_total * w
    # delimiters: one after seq and one after count per sequence => 2 * w * n_sequences
    bits_delimiters = 2 * w * n_sequences
    bits_counts = count_width * n_sequences
    total_bits = bits_events + bits_delimiters + bits_counts

    stats = {
        "bits_events": bits_events,
        "bits_delimiters": bits_delimiters,
        "bits_counts": bits_counts,
        "total_bits": total_bits,
        "n_sequences": n_sequences,
        "n_events_total": n_events_total
    }

    # Print requested stats (human readable)
    print(f"[serialize_sequences] sequences (events only) bits: {bits_events}")
    print(f"[serialize_sequences] delimiters bits:               {bits_delimiters}")
    print(f"[serialize_sequences] counts bits:                   {bits_counts}")
    print(f"[serialize_sequences] total bits:                    {total_bits}")
    print(f"[serialize_sequences] w (bits/event):                {w}")
    print(f"[serialize_sequences] count_width (bits/count):      {count_width}")
    print(f"[serialize_sequences] n_sequences:                   {n_sequences}")
    print(f"[serialize_sequences] n_events_total:                {n_events_total}")

    return bitstream, count_width, w, stats




def bits_to_bytes(bitstring: str) -> bytes:
    """Convert bitstring ('0101...') to bytes (8 bits per byte)."""
    if len(bitstring) % 8 != 0:
        raise ValueError(f"Bitstring length {len(bitstring)} is not byte-aligned")
    return bytes(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))

def bytes_to_bits(b) -> str:
    if isinstance(b, bytearray):
        b = bytes(b)
    return "".join(f"{byte:08b}" for byte in b)

import reedsolo


def ecc_encode(data_bits: str, nsym: int = 10) -> tuple[str, int]:
    orig_len = len(data_bits)
    pad_len = (8 - (orig_len % 8)) % 8
    data_bits_padded = data_bits + "0" * pad_len

    data_bytes = bits_to_bytes(data_bits_padded)
    rs = reedsolo.RSCodec(nsym)
    encoded_bytes = rs.encode(data_bytes)
    encoded_bits = bytes_to_bits(encoded_bytes)

    # ---- normalize to bitstring ----
    if not isinstance(encoded_bits, str):
        encoded_bits = "".join("1" if int(x) else "0" for x in encoded_bits)

    return encoded_bits, orig_len



# def spread_across_parameters(data_bits, model_weights, n_bits, seed=1234):
#     """
#     Spread the bitstream across parameters using a PRNG.
#     Returns a mapping: {param_name: [(index, bits), ...]}.
#     """
#     import random
#     random.seed(seed)
    
#     # Flatten all parameters into a big list of slots
#     slots = []

def _evaluate_as_attacker_adapted(
    *,
    defended_sd: Dict[str, torch.Tensor],
    train_source: Path,
    attacker_n_bits: int,
    nsym: int,
    delim_bits: str,
    options: Dict[str, Any],
    out_dir: Path,
    debug_k: int = 2,
    subseq_len: int = 3,   # kept for your own debugging use if needed later
    eval_top_k: Optional[int] = None,  # optional: evaluate only most frequent uniques
) -> Dict[str, Any]:
    """
    Adapted attacker view (ECC + serialization):
      - reconstruct payload_bits via dedup + serialize (GT)
      - pack_ecc_stream to model capacity
      - decode ECC bits from defended model
      - RS unpack -> recovered_plain (plaintext bits)
      - deserialize recovered_plain -> decoded unique seqs + counts
      - report SAME base similarity keys as baseline:
          bit_similarity, event_similarity, sequence_similarity, and membership rates
        but because order is absent, event/sequence similarity are defined via:
          - weighted recovery (occurrences) and unique recovery (patterns)
        We set base keys to WEIGHTED (baseline-comparable) and also report *_unique.
    """
    delim_token = int(delim_bits, 2)  # may be used in your debug pipelines

    # --- GT payload bits ---
    sequences = load_training_sequences(str(train_source))
    gt_unique, gt_counts = deduplicate_and_sort_sequences(sequences)

    payload_bits, count_width, w, ser_stats = serialize_sequences(
        gt_unique, gt_counts, num_classes=options["num_classes"]
    )
    payload_bits_total = len(payload_bits)

    # --- capacity (attacker) ---
    attacker_capacity_bits = count_capacity_bits(defended_sd, attacker_n_bits)

    # --- pack ECC to capacity ---
    ecc_stream, used_plain, overhead_bits, overhead_ratio, header_spans = pack_ecc_stream(
        payload_bits,
        nsym=int(nsym),
        capacity_bits=int(attacker_capacity_bits),
    )
    ecc_len = len(ecc_stream)

    # --- decode ECC bits from defended model ---
    rem = ecc_len % attacker_n_bits
    padded_bits_written = ecc_len + (attacker_n_bits - rem if rem else 0)

    exfil_padded = decode_all_parameters(defended_sd, attacker_n_bits, padded_bits_written)
    exfil_ecc = exfil_padded[:ecc_len]

    recovered_plain, rs_metrics = unpack_ecc_stream(exfil_ecc, nsym=int(nsym))
    intended_plain = payload_bits[:used_plain]
    recovered_plain = recovered_plain[:used_plain]

    # --- base bit similarity (always well-defined) ---
    bit_sim, bit_errs, bit_compared = bit_similarity(recovered_plain, intended_plain)

    # --- deserialize decoded payload ---
    decoded_unique, decoded_counts, _, parse_metrics = deserialize_sequences(
        recovered_plain,
        count_width=count_width,
        w=w,
        num_classes=options["num_classes"],
        strict=False,
    )

    # optional: restrict evaluation to top-k most frequent GT uniques
    if eval_top_k is not None:
        k = min(int(eval_top_k), len(gt_unique))
        gt_unique_eval = gt_unique[:k]
        gt_counts_eval = gt_counts[:k]
    else:
        gt_unique_eval = gt_unique
        gt_counts_eval = gt_counts

    # --- define sets/maps for recovery metrics ---
    gt_set = {tuple(s) for s in gt_unique_eval}
    dec_set = {tuple(s) for s in decoded_unique}
    gt_map = {tuple(s): int(c) for s, c in zip(gt_unique_eval, gt_counts_eval)}

    # membership rates over UNIQUE sequences/events (pattern coverage)
    # (we reuse the membership helper by turning uniques into lists)
    seq_mem_unique, evt_mem_unique, mem_counts = _membership_rates(
        _adapted_to_sequence_lists_for_membership(gt_unique_eval),
        _adapted_to_sequence_lists_for_membership(decoded_unique),
        known_events=options.get("num_classes"),
    )

    # unique “exact” sequence similarity (pattern recovery)
    seq_sim_unique = _unique_exact_rates_from_sets(gt_set, dec_set)

    # weighted recovery (baseline-comparable, occurrence-weighted)
    seq_sim_weighted, evt_sim_weighted, w_counts = _weighted_recovery_from_maps(gt_map, dec_set)

    # Choose BASE keys to be weighted (closest to baseline raw-stream semantics)
    seq_sim_base = seq_sim_weighted
    evt_sim_base = evt_sim_weighted
    seq_mem_base = seq_sim_unique  # or seq_mem_unique; both are set-overlap rates
    evt_mem_base = evt_mem_unique

    # Debug dump: top sequences (by GT frequency) vs decoded (first few)
    debug_path = out_dir / "debug_unique_sequences.txt"
    try:
        with open(debug_path, "w") as f:
            f.write(f"w(bits/event)={w}, count_width={count_width}, delim_bits={delim_bits}\n")
            f.write(f"RS metrics: {rs_metrics}\n")
            f.write(f"Parse metrics: {parse_metrics}\n\n")

            f.write("=== GT uniques (top) ===\n")
            for i in range(min(debug_k, len(gt_unique_eval))):
                f.write(f"GT #{i} | count={gt_counts_eval[i]} | {' '.join(map(str, gt_unique_eval[i]))}\n")

            f.write("\n=== Decoded uniques (first) ===\n")
            for i in range(min(debug_k, len(decoded_unique))):
                f.write(f"DE #{i} | count={decoded_counts[i] if i < len(decoded_counts) else 'NA'} | "
                        f"{' '.join(map(str, decoded_unique[i]))}\n")

            f.write("\n=== Plain bits head ===\n")
            f.write("INTENDED_HEAD: " + intended_plain[:256] + "\n")
            f.write("RECOVERED_HEAD: " + recovered_plain[:256] + "\n")
    except Exception:
        debug_path = None

    dataset_coverage = (used_plain / payload_bits_total) if payload_bits_total else 0.0
    model_utilization = (used_plain / attacker_capacity_bits) if attacker_capacity_bits else 0.0

    return {
        "attack_context": {
            "approach": "adapted",
            "attacker_n_bits": int(attacker_n_bits),
            "nsym": int(nsym),
            "delim_bits": str(delim_bits),
        },
        "serialization": ser_stats,
        "capacity": {
            "model_capacity_bits": int(attacker_capacity_bits),
            "payload_bits_total": int(payload_bits_total),
            "plain_bits_used": int(used_plain),
            "dataset_coverage": float(dataset_coverage),
            "model_utilization": float(model_utilization),

            "ecc_bits_total": int(ecc_len),
            "padded_bits_written": int(padded_bits_written),
            "ecc_overhead_bits": int(overhead_bits),
            "ecc_overhead_ratio": float(overhead_ratio),
            "num_headers": int(len(header_spans)),
        },
        "decoding": {
            "rs": rs_metrics,
            "parse": parse_metrics,
        },
        "similarity": {
            # base similarity metrics (same keys as baseline)
            "bit_similarity": float(bit_sim),
            "event_similarity": float(evt_sim_base),
            "sequence_similarity": float(seq_sim_base),
            "event_membership_rate": float(evt_mem_base),
            "sequence_membership_rate": float(seq_mem_base),

            # shared counts
            "bits_compared": int(bit_compared),
            "bit_errors": int(bit_errs),

            # extra (optional) useful metrics for adapted (comparable + robust)
            "event_similarity_weighted": float(evt_sim_weighted),
            "sequence_similarity_weighted": float(seq_sim_weighted),
            "event_similarity_unique": float(evt_mem_unique),          # unique events coverage
            "sequence_similarity_unique": float(seq_sim_unique),       # unique sequence pattern recovery
            "sequence_membership_rate_unique": float(seq_mem_unique),  # explicit

            # membership counts
            **mem_counts,
            # weighted totals/hits
            **w_counts,
        },
        "paths": {
            **({"debug_unique_sequences": str(debug_path)} if debug_path else {})
        }
    }

#     for name, tensor in model_weights.items():
#         slots.extend([(name, idx) for idx in range(tensor.numel())])
    
#     random.shuffle(slots)
    
#     mapping = {}
#     bit_idx = 0
#     for name, idx in slots:
#         if bit_idx >= len(data_bits):
#             break
#         mapping.setdefault(name, []).append((idx, data_bits[bit_idx:bit_idx+n_bits]))
#         bit_idx += n_bits
    
#     return mapping
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def params_available(model_weights: dict) -> int:
    return sum(t.numel() for t in model_weights.values())


def select_least_sensitive_params(model_weights):
    """
    Placeholder: currently returns all params.
    Later could filter by layer type, gradient stats, etc.
    """
    return model_weights


import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

def _in_any_span(x: int, spans):
    for s, e in spans:
        if s <= x < e:
            return True
    return False

def _top_lsb_indices(n_bits: int, header_top_k: int):
    return [n_bits - 1 - i for i in range(header_top_k)]

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

def insert_data_smart(
    target_params: Dict[str, torch.Tensor],
    spread_mapping: Dict[str, List[Tuple[int, str]]],
    n_bits: int,
    *,
    slots: Optional[List[Tuple[str, int]]] = None,     # kept for API compatibility (unused here)
    header_spans: Optional[List[Tuple[int, int]]] = None,  # ignored in header-less framing
    header_top_k: int = 0                                   # ignored in header-less framing
) -> Dict[str, torch.Tensor]:
    """
    Header-less embedding:
      For each mapping entry (param_name, flat_index, chunk_bits_len_n_bits),
      write chunk[j] into LSBit index j (0..n_bits-1) of that uint32 word.

    Assumes `bits_to_spread_mapping` produced chunks of length exactly `n_bits`.
    Extraction must read bits back in the same order (0..n_bits-1).
    """
    new_weights: Dict[str, torch.Tensor] = {}

    # Build uint32 views for float32 tensors
    views: Dict[str, np.ndarray] = {}
    for name, t in target_params.items():
        if t.dtype == torch.float32:
            views[name] = t.detach().cpu().numpy().view(np.uint32)

    for name, t in target_params.items():
        u32 = views.get(name)
        if u32 is None:
            # non-float tensors: passthrough
            new_weights[name] = t.detach().clone()
            continue

        entries = spread_mapping.get(name, [])
        if not entries:
            new_weights[name] = t.detach().clone()
            continue

        for idx, chunk in entries:
            if idx >= u32.size:
                continue
            val = int(u32.flat[idx])

            # write up to n_bits (ignore any extra, if mapping ever overshoots)
            L = min(len(chunk), n_bits)
            for j in range(L):
                b = 1 if chunk[j] == "1" else 0
                # clear LSBit j, then set it
                val = (val & ~(1 << j)) | (b << j)

            u32.flat[idx] = np.uint32(val)

        # write back
        new_weights[name] = torch.from_numpy(u32.view(np.float32)).to(t.device)

    # copy any tensors we didn't touch
    for name, t in target_params.items():
        if name not in new_weights:
            new_weights[name] = t.detach().clone()

    return new_weights


"""
DECODING
"""

def _ensure_bitstring(bits) -> str:
    """Convert list/tuple/np array of 0/1 to '0101...'."""
    if isinstance(bits, str):
        return bits
    if isinstance(bits, (bytes, bytearray)):
        # treat as raw bytes -> bits
        return "".join(f"{b:08b}" for b in bits)
    if isinstance(bits, (list, tuple)):
        return "".join("1" if int(x) else "0" for x in bits)
    # numpy array or other iterable
    try:
        return "".join("1" if int(x) else "0" for x in bits)
    except Exception as e:
        raise TypeError(f"Unsupported bits type {type(bits)}") from e



def ecc_decode(encoded_bits: str, orig_len: int, nsym: int = 10) -> str:
    """
    Decode RS-encoded bitstring and return the recovered *original* bitstring of length orig_len.
    Robust to varying reedsolo return types and to bytes_to_bits returning list[int].
    """
    encoded_bits = _ensure_bitstring(encoded_bits)

    # pad to full bytes
    pad_len = (-len(encoded_bits)) % 8
    if pad_len:
        encoded_bits = encoded_bits + ("0" * pad_len)

    encoded_bytes = bits_to_bytes(encoded_bits)  # your helper
    rs = reedsolo.RSCodec(nsym)

    decoded = rs.decode(encoded_bytes)

    # reedsolo may return:
    # - bytes-like message
    # - tuple(msg, ecc, errata_pos) OR tuple(msg, ecc)
    if isinstance(decoded, tuple):
        decoded_bytes = decoded[0]
    else:
        decoded_bytes = decoded

    # normalize to 'bytes'
    if not isinstance(decoded_bytes, (bytes, bytearray)):
        decoded_bytes = bytes(decoded_bytes)

    decoded_bits = bytes_to_bits(decoded_bytes)  # your helper (may return str OR list)
    decoded_bits = _ensure_bitstring(decoded_bits)

    return decoded_bits[:orig_len]


def despread_from_parameters(model_weights, spread_mapping, n_bits):
    """
    Collect bits from model weights using the same spread_mapping as encoder.
    Returns concatenated bitstream.
    """
    bitstream = ""
    for name, param in model_weights.items():
        if name in spread_mapping:
            flat_tensor = param.flatten()
            for idx, _ in spread_mapping[name]:
                val = flat_tensor[idx].item()
                bin_val = float_to_bin32_le(val)
                bitstream += bin_val[-n_bits:]
    return bitstream

from typing import List, Tuple
from typing import List, Tuple, Dict, Any, Optional

def deserialize_sequences(
    bitstream: str,
    count_width: int,
    w: int,
    num_classes: int,
    *,
    strict: bool = False,
) -> Tuple[List[List[int]], List[int], int, Dict[str, Any]]:
    """
    Decode stream produced by serialize_sequences():

      (events: w-bit each) [delim=w ones] (count: fixed width, multiple of w) [delim]

    Args:
      bitstream: serialized '0'/'1' string
      count_width: bits per count (MUST be known; do NOT infer via delimiters)
      w: bits per event
      num_classes: number of valid event classes (events are 0..num_classes-1 in stream)
      strict: if True, raise on corruption; if False, stop gracefully

    Returns:
      unique_sequences (1-based events), counts, count_width, metrics
    """
    if w <= 0:
        raise ValueError("w must be > 0")
    if count_width <= 0 or (count_width % w) != 0:
        raise ValueError(f"count_width must be >0 and multiple of w. got count_width={count_width}, w={w}")

    delim = "1" * w
    n = len(bitstream)
    i = 0

    unique_sequences: List[List[int]] = []
    counts: List[int] = []
    cur_seq_0: List[int] = []

    stopped_on_incomplete_tail = False
    stop_reason = "eof"
    invalid_event_tokens = 0
    delims_seen = 0
    events_parsed = 0

    # only parse full w-bit tokens
    n_aligned = (n // w) * w
    if n_aligned != n:
        # ignore non-aligned tail bits
        n = n_aligned
        stopped_on_incomplete_tail = True
        stop_reason = "non_aligned_tail"

    while i + w <= n:
        token = bitstream[i:i+w]
        i += w

        if token != delim:
            e = int(token, 2)
            if e >= num_classes:
                invalid_event_tokens += 1
                if strict:
                    raise ValueError(f"decoded event code {e} ≥ num_classes={num_classes}")
                stopped_on_incomplete_tail = True
                stop_reason = "event_out_of_range"
                break
            cur_seq_0.append(e)
            events_parsed += 1
            continue

        # token == delim: sequence boundary
        delims_seen += 1

        # Need fixed-width count
        if i + count_width > n:
            stopped_on_incomplete_tail = True
            stop_reason = "incomplete_count_bits"
            break

        count_bits = bitstream[i:i+count_width]
        i += count_width
        c = int(count_bits, 2)

        # Require trailing delimiter after count
        if i + w > n:
            stopped_on_incomplete_tail = True
            stop_reason = "missing_count_delim"
            break

        tail = bitstream[i:i+w]
        if tail != delim:
            if strict:
                raise ValueError("Malformed stream: missing delimiter after count")
            stopped_on_incomplete_tail = True
            stop_reason = "count_delim_mismatch"
            break
        i += w
        delims_seen += 1

        unique_sequences.append([e + 1 for e in cur_seq_0])
        counts.append(c)
        cur_seq_0 = []

    metrics = {
        "parse_ok": (not strict) or (stop_reason in ("eof",)),
        "stop_reason": stop_reason,
        "stopped_on_incomplete_tail": stopped_on_incomplete_tail,
        "sequences_parsed": len(unique_sequences),
        "events_parsed": events_parsed,
        "delims_seen": delims_seen,
        "invalid_event_tokens": invalid_event_tokens,
        "bits_consumed": i,
        "bits_total_aligned": n,
    }

    return unique_sequences, counts, count_width, metrics

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


from pathlib import Path
import torch

from pathlib import Path
import torch


def save_attacked_model(new_weights, cfg, options, n_bits, approach):
    """
    Save attacked checkpoint in structured format:

    models/attacked/white_box/<dataset>/<split>/
        <dataset>__atk=lsb(n{n_bits},{approach},compressed={0|1})/
            ckpt_attacked.pth
    """

    dataset_id = cfg.get("dataset", {}).get("name", "dataset")
    target_id = cfg.get("experiment", {}).get("target_id", "split0")

    pre = cfg.get("attack", {}).get("preprocessing", {}) or {}
    huffman_enabled = bool(pre.get("huffman", False))

    folder_name = (
        f"{dataset_id}"
        f"__atk=lsb(n{n_bits},{approach},compressed={int(huffman_enabled)})"
    )

    out_dir = (
        Path("models/attacked/white_box")
        / dataset_id
        / target_id
        / folder_name
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    attacked_ckpt_path = out_dir / "ckpt_attacked.pth"

    # keep predictor-compatible format
    torch.save({"state_dict": new_weights}, attacked_ckpt_path)

    return out_dir, attacked_ckpt_path


def evaluate_model(Model, options, attacked_ckpt_path):
    options_for_pred = dict(options)
    options_for_pred["model_path"] = str(attacked_ckpt_path)
    predicter = Predicter(Model, options_for_pred)
    P, R, F1, FP, FN, prediction_time = predicter.predict_unsupervised(return_metrics=True)
    return P, R, F1, FP, FN, prediction_time


def write_bitstreams(out_dir, original_bits, decoded_bits):
    with open(out_dir / 'bitstream_original.txt', 'w') as f:
        f.write(original_bits.strip())
    with open(out_dir / 'bitstream_decoded.txt', 'w') as f:
        f.write(decoded_bits.strip())

    return hamming_distance_from_files(
        str(out_dir / 'bitstream_original.txt'),
        str(out_dir / 'bitstream_decoded.txt')
    )

from collections import Counter

def verify_decoding_smart(decoded_u, decoded_c, orig_u, orig_c,
                                save_path_prefix="decoded_output",
                                known_events=33, subseq_len=3):
    """
    Compare unique sequences + counts (ignores order of full dataset).
    """
    # dicts: sequence -> count
    orig_dict = {tuple(seq): c for seq, c in zip(orig_u, orig_c)}
    dec_dict  = {tuple(seq): c for seq, c in zip(decoded_u, decoded_c)}

    # sets of all unique events
    orig_events = {e for seq in orig_u for e in seq}
    dec_events  = {e for seq in decoded_u for e in seq}

    # sequence membership (how many orig unique seqs are in decoded)
    seq_membership = sum(1 for s in orig_dict if s in dec_dict)

    # event membership
    ship_count = len(orig_events & dec_events)
    event_membership_total = min(known_events, len(orig_events))

    # count matches and distance
    count_matches = sum(1 for s in orig_dict if s in dec_dict and orig_dict[s] == dec_dict[s])
    count_distance = sum(abs(orig_dict.get(s,0) - dec_dict.get(s,0))
                         for s in set(orig_dict) | set(dec_dict))

    # subsequence matches (optional, over unique sequences)
    subseq_match = 0
    subseq_total = 0
    if subseq_len is not None:
        for seq in orig_u:
            for i in range(len(seq) - subseq_len + 1):
                orig_sub = tuple(seq[i:i+subseq_len])
                subseq_total += 1
                if any(tuple(dseq[j:j+subseq_len]) == orig_sub
                       for dseq in decoded_u
                       for j in range(len(dseq)-subseq_len+1)):
                    subseq_match += 1

    print(f"\n📊 Verification metrics:")
    print(f"   🧩 Sequence membership     : {seq_membership}/{len(orig_u)} ({seq_membership/len(orig_u):.2%})")
    print(f"   🎯 Event membership        : {event_membership_count}/{event_membership_total} ({event_membership_count/event_membership_total:.2%})")
    print(f"   🔢 Count matches           : {count_matches}/{len(orig_u)} ({count_matches/len(orig_u):.2%})")
    print(f"   📉 Count distance (L1)     : {count_distance}")

    if subseq_len is not None:
        rate = subseq_match / subseq_total if subseq_total else 0
        print(f"   🔹 Subsequence matches (len={subseq_len}): {subseq_match}/{subseq_total} ({rate:.2%})")

    # Save unique sequences + counts for inspection
    orig_file = f"{save_path_prefix}_original.txt"
    dec_file  = f"{save_path_prefix}_decoded.txt"
    with open(orig_file, "w") as f:
        for seq, c in zip(orig_u, orig_c):
            f.write(f"{' '.join(map(str,seq))} | {c}\n")
    with open(dec_file, "w") as f:
        for seq, c in zip(decoded_u, decoded_c):
            f.write(f"{' '.join(map(str,seq))} | {c}\n")

    print(f"\n📝 Files saved:")
    print(f"   📄 Original unique seqs → {orig_file}")
    print(f"   📄 Decoded unique seqs  → {dec_file}")

    return {
        "sequence_membership": seq_membership/len(orig_u),
        "event_membership": event_membership_count/event_membership_total,
        "event_membership_count": event_membership_count,
        "event_membership_total": event_membership_total,
        "count_matches": count_matches,
        "count_coverage": count_matches/len(orig_u),
        "count_distance": count_distance,
        "subsequence_coverage": subseq_match/subseq_total,
    }

# try #2 streaming instead of blocks
def u32(n: int) -> str:
    if n < 0 or n >= (1<<32): raise ValueError("u32 out of range")
    return format(n, "032b")

from typing import List, Tuple

def pack_ecc_stream(payload_bits: str,
                    nsym: int,
                    capacity_bits: int,
                    probe_min_plain: int = 1024
                   ) -> Tuple[str, int, int, float, List[Tuple[int, int]]]:
    """
    Build: [orig_len(32)][enc_len(32)][enc_bits] ... until capacity reached.
    Greedy: at each step, binary-search largest plaintext slice that fits remaining capacity.

    Returns:
        ecc_stream: concatenated bitstring
        used_plain: number of plaintext (useful) bits consumed from payload_bits
        overhead_bits: (headers + ECC parity) consumed
        overhead_ratio: overhead_bits / total_bits_used
        header_spans: list of (start_bit, end_bit) half-open intervals for each 64-bit header
                      measured over ecc_stream (0-based bit indices)
    """
    out: List[str] = []
    cursor = 0               # plaintext bits consumed
    used = 0                 # total bits written (headers + enc_bits)
    header_bits = 64
    header_spans: List[Tuple[int, int]] = []

    # Track current stream length in bits without repeatedly joining
    stream_len = 0

    def enc_of(L: int):
        chunk = payload_bits[cursor:cursor+L]
        enc_bits, orig_len = ecc_encode(chunk, nsym=nsym)  # must return (encoded_bits, L_used)
        return enc_bits, orig_len

    while used + header_bits < capacity_bits and cursor < len(payload_bits):
        cap_left = capacity_bits - used - header_bits
        max_plain = len(payload_bits) - cursor

        # Probe minimum
        Lmin = min(probe_min_plain, max_plain) if max_plain > 0 else 0
        if Lmin == 0:
            break

        enc_bits_min, _ = enc_of(Lmin)
        if len(enc_bits_min) > cap_left:
            # Need smaller than probe_min → binary search 1..Lmin
            lo, hi, best = 1, Lmin, 0
            best_enc = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                enc_bits_mid, _ = enc_of(mid)
                if len(enc_bits_mid) <= cap_left:
                    best = mid
                    best_enc = enc_bits_mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best == 0:
                # cannot fit any ECC chunk
                break
            enc_bits, orig_len = best_enc, best
        else:
            # Try to grow from Lmin..max_plain
            lo, hi, best = Lmin, max_plain, Lmin
            best_enc, _ = enc_of(Lmin)
            while lo <= hi:
                mid = (lo + hi) // 2
                enc_bits_mid, _ = enc_of(mid)
                if len(enc_bits_mid) <= cap_left:
                    best = mid
                    best_enc = enc_bits_mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            enc_bits, orig_len = best_enc, best

        # Write header: record its span on the *final stream*
        hdr_start = stream_len
        h1 = u32(orig_len)          # 32-bit MSB-first length
        h2 = u32(len(enc_bits))     # 32-bit MSB-first encoded length
        out.append(h1); out.append(h2)
        stream_len += header_bits
        header_spans.append((hdr_start, hdr_start + header_bits))

        # Write payload codeword bits
        out.append(enc_bits)
        stream_len += len(enc_bits)

        # Bookkeeping
        cursor += orig_len
        used += header_bits + len(enc_bits)

        # (overhead metrics are recomputed after loop)

    overhead_bits = used - cursor
    overhead_ratio = (overhead_bits / used) if used else 0.0

    return "".join(out), cursor, overhead_bits, overhead_ratio, header_spans

from typing import Tuple, Dict, Any

def unpack_ecc_stream(bitstream: str, nsym: int) -> Tuple[str, Dict[str, Any]]:
    """
    Robust: never raises if headers are zeroed/corrupt; returns "" + zeroed metrics with flags.
    Frame: [orig_len(32)][enc_len(32)][enc_bits]...
    """
    n = len(bitstream)
    i = 0
    out = []
    per_chunk = []

    total_chunks = 0
    decode_ok = 0
    total_orig_bits = 0
    total_decoded_bits = 0
    total_enc_bits = 0
    total_header_bits = 0

    # Fast path: empty or all-zero stream
    if n == 0 or set(bitstream) == {"0"}:
        metrics = {
            "chunks_total": 0,
            "chunks_decoded_ok": 0,
            "decode_success_rate": 0.0,
            "useful_bits_intended": 0,
            "useful_bits_decoded": 0,
            "dataset_coverage_decoded": 0.0,
            "raw_bits_used": 0,
            "enc_bits_total": 0,
            "header_bits_total": 0,
            "protocol_overhead_bits_total": 0,
            "overhead_ratio": 0.0,
            "wasted_bits_total": 0,
            "per_chunk": [],
            "stopped_on_incomplete_tail": False,
            "stream_bits_consumed": 0,
            "stream_zeroed": True,
            "reason": "all_zero_or_empty_stream",
        }
        return "", metrics

    while i + 64 <= n:
        # Read header
        orig_len = int(bitstream[i:i+32], 2); i += 32
        enc_len  = int(bitstream[i:i+32], 2); i += 32
        total_header_bits += 64

        # Detect zeroed/corrupt header and exit cleanly
        if orig_len == 0 and enc_len == 0:
            # Roll back to start of header for accurate consumed pointer if you prefer:
            i -= 64
            break

        # Sanity checks (no exceptions)
        cap_left = n - i
        if enc_len <= 0 or enc_len > cap_left:
            # Incomplete/invalid tail — roll back the header read and stop
            i -= 64
            total_header_bits -= 64
            break

        # Need full codeword present
        enc_bits = bitstream[i:i+enc_len]; i += enc_len

        total_chunks += 1
        total_orig_bits += orig_len
        total_enc_bits += enc_len

        # Try decode; never raise
        try:
            dec_bits = ecc_decode(enc_bits, orig_len, nsym=nsym)
            out.append(dec_bits)
            total_decoded_bits += len(dec_bits)
            decode_ok += 1
            ok = True; reason = "ok"
        except Exception as e:
            ok = False; reason = str(e)

        # Protocol overhead (headers + parity beyond orig_len)
        overhead_bits = max(0, (enc_len + 64) - orig_len)
        per_chunk.append({
            "orig_len": orig_len,
            "enc_len": enc_len,
            "overhead_bits": overhead_bits,
            "overhead_ratio": (overhead_bits / (enc_len + 64)) if (enc_len + 64) else 0.0,
            "decode_success": ok,
            "reason": reason,
        })

    used_bits = total_enc_bits + total_header_bits
    protocol_overhead_bits_total = max(0, used_bits - total_orig_bits)
    wasted_bits_total = max(0, used_bits - total_decoded_bits)
    tail_incomplete = (i < n)
    decode_success_rate = (decode_ok / total_chunks) if total_chunks else 0.0
    dataset_coverage_decoded = (total_decoded_bits / total_orig_bits) if total_orig_bits else 0.0
    overhead_ratio = (protocol_overhead_bits_total / used_bits) if used_bits else 0.0

    # If we stopped immediately on a zeroed/corrupt header, signal it explicitly
    stream_zeroed = (total_chunks == 0 and total_header_bits == 0 and i < n)

    metrics = {
        "chunks_total": total_chunks,
        "chunks_decoded_ok": decode_ok,
        "decode_success_rate": decode_success_rate,
        "useful_bits_intended": total_orig_bits,
        "useful_bits_decoded": total_decoded_bits,
        "dataset_coverage_decoded": dataset_coverage_decoded,
        "raw_bits_used": used_bits,
        "enc_bits_total": total_enc_bits,
        "header_bits_total": total_header_bits,
        "protocol_overhead_bits_total": protocol_overhead_bits_total,
        "overhead_ratio": overhead_ratio,
        "wasted_bits_total": wasted_bits_total,
        "per_chunk": per_chunk,
        "stopped_on_incomplete_tail": tail_incomplete,
        "stream_bits_consumed": i,
    }
    if stream_zeroed:
        metrics["stream_zeroed"] = True
        metrics["reason"] = "headers_zeroed_or_invalid"

    return "".join(out), metrics




def collect_slots(target_params: dict, include_names=None) -> list:
    """
    Build a flat list of slots (param_name, flat_index).
    target_params: dict {name: tensor}
    include_names: list of substrings; only params whose name contains one will be used.
                   if None → all params.
    """
    slots = []
    for name, tensor in target_params.items():
        if include_names and not any(k in name for k in include_names):
            continue
        n = tensor.numel()
        for idx in range(n):
            slots.append((name, idx))
    return slots


# slots: flat list of (param_name, flat_index) giving write order

def bits_to_spread_mapping(
    bits: str,
    slots: List[Tuple[str, int]],
    n_bits: int,
    header_spans: Optional[List[Tuple[int,int]]] = None,
    header_replicas: int = 1,
    header_top_k: int = 1,
    header_replica_stride: Optional[int] = None,
) -> Tuple[Dict[str, List[Tuple[int, str]]], int]:
    """
    Map a bitstream to per-parameter chunks of length n_bits, where chunk[j] is written
    to LSBit index j of the (param, idx).
    Header bits (given by header_spans) are placed in the top LSBits (n_bits-1, n_bits-2, ...).
    Header is also replicated across different slots (header_replicas times).
    Payload bits fill the remaining positions (lowest LSBits upward).

    Returns: (spread, used_bits)
      spread: {param_name: [(flat_index, chunk_bits_len_n_bits), ...]}
    """
    cap_bits = len(slots) * n_bits
    if len(bits) > cap_bits:
        bits = bits[:cap_bits]

    # Build a mask for header positions in the stream for O(1) lookup
    header_mask = np.zeros(len(bits), dtype=np.bool_) if header_spans else None
    if header_spans:
        for s, e in header_spans:
            s = max(0, s); e = min(len(bits), e)
            if e > s:
                header_mask[s:e] = True

    # Precompute LSBit indices reserved for header (top of channel)
    # Example: header_top_k=1 -> [n_bits-1]; =2 -> [n_bits-1, n_bits-2]
    top_positions = list(range(n_bits-1, n_bits-1-header_top_k, -1))
    payload_positions = [i for i in range(n_bits) if i not in top_positions]

    # Plan header replica target slot indices (even spacing)
    total_slots = len(slots)
    if header_replica_stride is None and header_replicas > 1:
        header_replica_stride = max(1, total_slots // max(1, header_replicas))
    if header_replicas <= 0:
        header_replicas = 1

    spread = defaultdict(list)

    # Helper iterators for header & payload slot assignment
    slot_it = iter(range(total_slots))
    # For replicas, we’ll compute starting slot offsets
    replica_starts = [0]
    if header_replicas > 1:
        if header_replica_stride is None:
            # auto even placement
            replica_starts = [int(i * total_slots / header_replicas) for i in range(header_replicas)]
        else:
            replica_starts = [i * header_replica_stride for i in range(header_replicas)]
        replica_starts = [min(s, total_slots-1) for s in replica_starts]

    # Build an array of chunks initialized to '0' * n_bits for each slot we actually use
    chunks = [list("0"*n_bits) for _ in range(total_slots)]  # mutable lists for speed

    # Assign bits
    for bit_idx, b in enumerate(bits):
        is_header = (header_mask is not None and header_mask[bit_idx])

        if is_header:
            # Place this header bit into top positions AND replicate across slots
            for r, start in enumerate(replica_starts):
                # compute target slot for this header bit
                slot_pos = start + bit_idx  # simple sequential placement per replica
                if slot_pos >= total_slots:
                    # wrap if you prefer; for now, stop placing replicas that exceed capacity
                    break
                # pick one of the top positions (round-robin across top_positions)
                lsb_pos = top_positions[bit_idx % len(top_positions)]
                chunks[slot_pos][lsb_pos] = b
        else:
            # payload bit → fill into next available slot’s lowest available payload position
            # Mapping: continuous sequence across slots and payload_positions
            # Compute a linear payload index (exclude headers in count)
            # Simple mapping: assign by rank among payload bits:
            # rank = (# of payload bits seen so far)
            # slot = rank // len(payload_positions), pos = rank % len(payload_positions)
            # But we don't want O(n) state; maintain a running counter:
            pass  # we'll fill this below

    # Second pass for payload using a running counter (avoids O(n) lookups)
    payload_rank = 0
    for bit_idx, b in enumerate(bits):
        if header_mask is not None and header_mask[bit_idx]:
            continue
        # place payload bit
        if payload_positions:
            slot_pos = payload_rank // len(payload_positions)
            if slot_pos >= total_slots:
                break  # out of capacity
            lsb_pos = payload_positions[payload_rank % len(payload_positions)]
            chunks[slot_pos][lsb_pos] = b
            payload_rank += 1
        else:
            # if header occupies all n_bits (rare), no payload positions left
            break

    # Build the spread dict in your original shape
    used_slots = min(total_slots, max(
        1,
        (payload_rank + len(payload_positions) - 1) // max(1, len(payload_positions))
    ))
    # Also consider the farthest slot we touched for headers
    if header_spans:
        max_header_slot = max((start + (header_spans[-1][1]-1)) for start in replica_starts) if replica_starts else 0
        used_slots = max(used_slots, min(total_slots, max_header_slot+1))

    for si in range(used_slots):
        name, idx = slots[si]
        spread[name].append((idx, "".join(chunks[si])))

    # Tidy & return
    for k in spread:
        spread[k].sort(key=lambda t: t[0])
    used_bits = min(len(bits), used_slots * n_bits)
    return dict(spread), used_bits

def extract_bits_sequential(
    weights: Dict[str, torch.Tensor],
    slots: List[Tuple[str,int]],
    n_bits: int,
    *,
    header_spans: Optional[List[Tuple[int,int]]] = None,
    header_top_k: int = 1,
) -> str:
    """
    Inverse of insert_data_smart: for each slot s and position j,
    read from the LSB index assigned by the same header/payload rule,
    and append to the stream in order j=0..n_bits-1.
    """
    header_spans = header_spans or []
    top_idxs = _top_lsb_indices(n_bits, header_top_k)
    low_idxs = [i for i in range(n_bits) if i not in top_idxs]

    # Build views
    import numpy as np, torch
    views: Dict[str, np.ndarray] = {}
    for name, t in weights.items():
        if t.dtype == torch.float32:
            views[name] = t.detach().cpu().numpy().view(np.uint32)

    # Build slot_id map like in writer
    slot_id = {(name, idx): s for s, (name, idx) in enumerate(slots)}

    out_bits = []
    for name, idx in slots:
        u32 = views.get(name)
        if u32 is None or idx >= u32.size:
            break
        val = int(u32.flat[idx])
        s = slot_id[(name, idx)]

        # generate the same sequence of LSB indices we used when writing
        low_ptr = 0
        top_ptr = 0
        for j in range(n_bits):
            gpos = s * n_bits + j
            if _in_any_span(gpos, header_spans):
                lsb = top_idxs[top_ptr % len(top_idxs)]
                top_ptr += 1
            else:
                if not low_idxs:
                    continue
                lsb = low_idxs[low_ptr % len(low_idxs)]
                low_ptr += 1
            out_bits.append("1" if ((val >> lsb) & 1) else "0")

    return "".join(out_bits)




# choose header bit indices inside 0..n_bits-1 (0=LSB)
def header_bit_indices_for_top(n_bits: int, header_n_bits: int = 1):
    # return highest header_n_bits indices within the attack channel
    # e.g., n_bits=8, header_n_bits=2 -> [7,6]
    return [n_bits - 1 - i for i in range(header_n_bits)]

def set_bit_in_uint32(u32_val: int, bit_value: int, lsb_index: int) -> int:
    mask = ~(1 << lsb_index) & 0xFFFFFFFF
    return (u32_val & mask) | ((bit_value & 1) << lsb_index)

def embed_header_into_slot(u32_arr: np.ndarray, slot_idx: int, header_bits: str, header_bit_indices: list):
    """
    u32_arr: flattened uint32 view of parameter tensor
    slot_idx: starting index in u32_arr where header first bit goes
    header_bits: string of '0'/'1' length L
    header_bit_indices: list of LSB indices where to write each header bit replica bitwise
                       (if you want to embed same header bits across multiple bit-positions,
                        you can loop header_bit_indices)
    Strategy: for j-th header bit, write it to u32_arr[slot_idx + j] at each lsb index in header_bit_indices
    """
    L = len(header_bits)
    for j in range(L):
        idx = slot_idx + j
        if idx >= len(u32_arr):
            raise IndexError("not enough capacity in this param slot")
        bit = int(header_bits[j])
        for lsb in header_bit_indices:
            u32_arr[idx] = set_bit_in_uint32(int(u32_arr[idx]), bit, lsb)

def _in_any_span(x: int, spans):
    # spans: list[(start,end)] half-open
    # If spans are numerous, you can binary-search; for now linear is fine.
    for s, e in spans:
        if s <= x < e:
            return True
    return False

def _top_lsb_indices(n_bits: int, header_top_k: int):
    # e.g., n_bits=8, header_top_k=1 -> [7]; =2 -> [7,6]
    return [n_bits - 1 - i for i in range(header_top_k)]


import numpy as np
import torch
from typing import List, Tuple, Dict

def build_uint32_views(params: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    views = {}
    for name, t in params.items():
        if t.dtype != torch.float32:
            continue
        arr = t.detach().cpu().numpy()
        views[name] = arr.view(np.uint32)
    return views

def write_header_bits_across_slots(
    views: Dict[str, np.ndarray],
    slots: List[Tuple[str, int]],
    start_slot: int,
    header_bits: str,
    header_bit_indices: List[int],
):
    """Write header_bits (len L) starting at slots[start_slot], one bit per slot,
       into each lsb index in header_bit_indices (replication across LSB positions)."""
    L = len(header_bits)
    for j in range(L):
        si = start_slot + j
        if si >= len(slots):
            raise IndexError("Not enough slots to place header.")
        name, idx = slots[si]
        u32 = views[name]
        val = int(u32.flat[idx])
        b = 1 if header_bits[j] == "1" else 0
        for lsb in header_bit_indices:
            val = set_bit_in_uint32(val, b, lsb)
        u32.flat[idx] = np.uint32(val)

def remove_spans(bits: str, spans: List[Tuple[int, int]]) -> str:
    """Return bits with all [start,end) spans removed (for payload-only stream)."""
    if not spans:
        return bits
    out = []
    cur = 0
    for s, e in sorted(spans):
        if cur < s:
            out.append(bits[cur:s])
        cur = e
    if cur < len(bits):
        out.append(bits[cur:])
    return "".join(out)


# ========================
# DEFENSE UTILS 
# =======================

from pathlib import Path
from typing import Dict, Callable, List, Optional
import torch
import numpy as np

def zero_low_bits_uint32_array(u32: np.ndarray, n_bits: int) -> np.ndarray:
    """Clear the lowest n_bits of a uint32 numpy array."""
    if n_bits <= 0:
        return u32
    if n_bits >= 32:
        return np.zeros_like(u32, dtype=np.uint32)
    mask = ~((1 << n_bits) - 1) & 0xFFFFFFFF
    return u32 & np.uint32(mask)

def sanitize_tensor_bits(tensor: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    For float32 tensors: reinterpret as uint32, zero lowest n_bits, back to float32.
    Non-float32 tensors are copied unchanged.
    """
    if not tensor.is_floating_point() or tensor.dtype != torch.float32:
        return tensor.clone()
    arr = tensor.detach().cpu().numpy()
    u32 = arr.view(np.uint32).copy()
    u32_sanit = zero_low_bits_uint32_array(u32, n_bits)
    sanitized = u32_sanit.view(np.float32)
    return torch.from_numpy(sanitized).to(tensor.device)

def sanitize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    n_bits: int,
    param_filter: Callable[[str], bool] = lambda n: True,
) -> Dict[str, torch.Tensor]:
    """Return a new state_dict with lowest n_bits zeroed for selected parameters."""
    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if param_filter(k) and isinstance(v, torch.Tensor):
            new_sd[k] = sanitize_tensor_bits(v, n_bits)
        else:
            new_sd[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return new_sd

def load_state_dict_from_file(path: Path) -> Dict[str, torch.Tensor]:
    """Load a PyTorch state_dict; unwrap 'state_dict' key if present."""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd

def save_defended_model(
    new_weights,
    cfg,
    options,
    n_bits: int,
    attacked_name: str,
    out_root: Path = Path("models/defended"),
):
    """
    Save a defended checkpoint in a structured folder:
    models/defended/{ds_name}__def=lsb(n{n_bits})__src={attacked_name}__run={run_id}/ckpt_defended.pth
    """
    ds_name = cfg.get("dataset", {}).get("name", "dataset")
    exp_name = cfg.get("experiment", {}).get("name", "wb_lsb")
    run_id = options.get("run_id", "auto")

    out_dir = (
        out_root
        / f"{ds_name}__def=lsb(n{n_bits})__src={attacked_name}__run={run_id}"
    )
    _ensure_dir(out_dir)

    defended_ckpt_path = out_dir / "ckpt_defended.pth"
    torch.save({"state_dict": new_weights}, defended_ckpt_path)
    return out_dir, defended_ckpt_path

def find_model_files_in_folder(folder: Path, extensions=(".pt", ".pth", ".bin")) -> List[Path]:
    files: List[Path] = []
    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))
    return files




# SIMPLE ATTACK (ONLY ECC)

import math
from typing import Tuple

def pack_ecc_stream_nohdr(payload_bits: str,
                          nsym: int,
                          capacity_bits: int,
                          orig_len: int = 1024,
                          symbol_bits: int = 8
                         ) -> Tuple[str, int, int, float, int, int]:
    """
    Return:
      ecc_stream, used_plain, overhead_bits, overhead_ratio, enc_len, orig_len
    """
    enc_parity = nsym * symbol_bits
    enc_len = orig_len + enc_parity

    out = []
    used_plain = 0
    used_bits = 0

    # number of chunks that fit in capacity
    max_chunks = capacity_bits // enc_len
    if max_chunks == 0:
        return "", 0, 0, 0.0, enc_len, orig_len

    # chunk loop
    for c in range(max_chunks):
        start = used_plain
        end = min(start + orig_len, len(payload_bits))
        plain = payload_bits[start:end]
        if not plain:
            break
        # pad last chunk to orig_len with zeros
        if len(plain) < orig_len:
            plain = plain + "0" * (orig_len - len(plain))

        enc_bits, _L = ecc_encode(plain, nsym=nsym)   # must output len(enc_bits)==enc_len
        # (if your ecc_encode returns variable length, left-pad parity to fixed enc_len)
        if len(enc_bits) != enc_len:
            raise ValueError(f"ecc_encode produced {len(enc_bits)} != enc_len {enc_len}")

        out.append(enc_bits)
        used_plain += min(orig_len, len(payload_bits) - start)
        used_bits  += enc_len

        if used_plain >= len(payload_bits):
            break

    overhead_bits = used_bits - used_plain
    overhead_ratio = (overhead_bits / used_bits) if used_bits else 0.0
    return "".join(out), used_plain, overhead_bits, overhead_ratio, enc_len, orig_len

from typing import Tuple, Dict, Any

def unpack_ecc_stream_nohdr(bitstream: str,
                            nsym: int,
                            enc_len: int,
                            orig_len: int,
                            used_plain: int = None) -> Tuple[str, Dict[str, Any]]:
    """
    Slice stream in fixed enc_len blocks, RS-decode each to orig_len.
    If used_plain is provided, trim the decoded tail to that many bits.
    """
    n = len(bitstream)
    chunks = n // enc_len
    out = []
    per_chunk = []
    decode_ok = 0

    for k in range(chunks):
        start = k * enc_len
        enc_bits = bitstream[start:start+enc_len]
        try:
            dec_bits = ecc_decode(enc_bits, orig_len, nsym=nsym)
            if len(dec_bits) != orig_len:
                dec_bits = dec_bits[:orig_len]  # be strict
            out.append(dec_bits)
            ok, reason = True, "ok"
            decode_ok += 1
        except Exception as e:
            ok, reason = False, str(e)
            # keep alignment: append zeros for failed chunk
            out.append("0" * orig_len)

        per_chunk.append({
            "chunk": k, "decode_success": ok, "reason": reason
        })

    recovered = "".join(out)
    if used_plain is not None:
        recovered = recovered[:used_plain]

    metrics = {
        "chunks_total": chunks,
        "chunks_decoded_ok": decode_ok,
        "decode_success_rate": (decode_ok / chunks) if chunks else 0.0,
        "useful_bits_intended": (orig_len * chunks if used_plain is None else used_plain),
        "useful_bits_decoded": len(recovered),
        "enc_bits_total": enc_len * chunks,
        "header_bits_total": 0,
        "per_chunk": per_chunk,
        "stream_zeroed": (n == 0 or set(bitstream) == {"0"}),
        "reason": "no_headers_fixed_block_framing",
    }
    return recovered, metrics



# simple version 3
# ============================================================
# Minimal ECC + LSB embed/extract for model parameters (PyTorch)
# ============================================================
from typing import Dict, Iterable, List, Tuple, Callable
import torch

# -------------------------
# Bit utilities
# -------------------------
def bits_from_str(s: str) -> List[int]:
    return [1 if c == "1" else 0 for c in s]

def bits_to_str(b: Iterable[int]) -> str:
    return "".join("1" if x else "0" for x in b)

def int_to_bits(n: int, width: int) -> List[int]:
    return [(n >> i) & 1 for i in reversed(range(width))]

def bits_to_int(b: Iterable[int]) -> int:
    v = 0
    for bit in b:
        v = (v << 1) | (1 if bit else 0)
    return v

# -------------------------
# Hamming(7,4) ECC (single-bit correction)
# Codeword bits order: [p1, p2, d1, p3, d2, d3, d4]
# -------------------------
def hamming74_encode_bits(payload_bits: List[int]) -> List[int]:
    out = []
    # pad to multiple of 4
    pad = (-len(payload_bits)) % 4
    payload_bits = payload_bits + [0]*pad
    for i in range(0, len(payload_bits), 4):
        d1, d2, d3, d4 = payload_bits[i:i+4]
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p3 = d2 ^ d3 ^ d4
        cw = [p1, p2, d1, p3, d2, d3, d4]
        out.extend(cw)
    return out

def hamming74_decode_bits(code_bits: List[int]) -> Tuple[List[int], int, int]:
    """
    Returns: (decoded_payload_bits, corrected_count, uncorrectable_count)
    Drops trailing pad nibble if present using length header upstream.
    """
    assert len(code_bits) % 7 == 0, "Hamming(7,4) stream must be multiple of 7 bits."
    decoded, corrected, uncorrectable = [], 0, 0
    for i in range(0, len(code_bits), 7):
        p1, p2, d1, p3, d2, d3, d4 = code_bits[i:i+7]
        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4
        syndrome = (s1 << 2) | (s2 << 1) | s3
        if syndrome:
            # positions 1..7 (p1,p2,d1,p3,d2,d3,d4)
            pos = syndrome  # 1-based
            if 1 <= pos <= 7:
                # flip bit
                idx = i + (pos - 1)
                code_bits[idx] ^= 1
                corrected += 1
            else:
                uncorrectable += 1
        # (possibly corrected) extract data bits
        _, _, d1, _, d2, d3, d4 = code_bits[i:i+7]
        decoded.extend([d1, d2, d3, d4])
    return decoded, corrected, uncorrectable

# -------------------------
# Length header + framing
# -------------------------
_HEADER_BITS = 32  # store original (pre-ECC) payload length in bits

def frame_with_len_header(payload_bits: List[int]) -> List[int]:
    """
    Add a fixed 32-bit big-endian header with original payload length (bits),
    then Hamming(7,4)-encode the whole framed stream (header+payload).
    """
    header = int_to_bits(len(payload_bits), _HEADER_BITS)
    framed = header + payload_bits
    return hamming74_encode_bits(framed)

def deframe_and_recover(encoded_bits: List[int]) -> Tuple[List[int], dict]:
    """
    Hamming-decodes, reads 32-bit length header, trims padding to exact length.
    Returns (original_bits, stats).
    """
    dec, corrected, uncorrectable = hamming74_decode_bits(encoded_bits[:])
    if len(dec) < _HEADER_BITS:
        raise ValueError("Decoded stream too short for header.")
    orig_len = bits_to_int(dec[:_HEADER_BITS])
    body = dec[_HEADER_BITS:]
    if len(body) < orig_len:
        raise ValueError("Decoded payload shorter than specified length.")
    body = body[:orig_len]
    stats = {
        "ecc_corrected_bits": corrected,
        "ecc_uncorrectable_blocks": uncorrectable,
        "orig_len_bits": orig_len,
    }
    return body, stats

# -------------------------
# Parameter selection & capacity
# -------------------------
def iter_param_tensors(state_dict: Dict[str, torch.Tensor],
                       include: Callable[[str, torch.Tensor], bool] = None):
    """
    Deterministic iteration of float32 tensors suitable for bit LSB embedding.
    Default: all float32 leaf tensors.
    """
    for name in sorted(state_dict.keys()):
        t = state_dict[name]
        if not isinstance(t, torch.Tensor):
            continue
        if include and not include(name, t):
            continue
        if t.dtype == torch.float32:
            yield name, t

def total_capacity_bits(state_dict: Dict[str, torch.Tensor],
                        n_lsb: int,
                        include: Callable[[str, torch.Tensor], bool] = None) -> int:
    cap = 0
    for _, t in iter_param_tensors(state_dict, include):
        cap += t.numel() * n_lsb
    return cap

# -------------------------
# Embed / Extract
# -------------------------
def embed_bits_in_params(state_dict: Dict[str, torch.Tensor],
                         payload_bits: List[int],
                         n_lsb: int = 1,
                         include: Callable[[str, torch.Tensor], bool] = None
                         ) -> Tuple[Dict[str, torch.Tensor], dict]:
    """
    Frames+ECC the payload, then writes into LSBs of float32 parameter tensors
    in a deterministic order. Requires enough capacity (kept simplest).
    """
    assert 1 <= n_lsb <= 8, "n_lsb must be in [1,8]"
    enc_bits = frame_with_len_header(payload_bits)
    need = len(enc_bits)
    cap = total_capacity_bits(state_dict, n_lsb, include)
    if need > cap:
        raise ValueError(f"Not enough capacity: need {need}, have {cap} (n_lsb={n_lsb}).")

    new_sd = {k: v.clone() if isinstance(v, torch.Tensor) else v
              for k, v in state_dict.items()}

    bit_idx = 0
    slots_used = 0
    for name, t in iter_param_tensors(new_sd, include):
        flat = t.view(-1).to(torch.float32)
        ints = flat.view(torch.int32)  # aliasing of float32 memory
        # process in chunks
        count = min(ints.numel(), (need - bit_idx + n_lsb - 1) // n_lsb)
        if count <= 0:
            break
        mask = ~((1 << n_lsb) - 1)
        vals = ints[:count]
        # build packed LSB int values from bits
        lsb_vals = torch.zeros(count, dtype=torch.int32)
        for i in range(count):
            v = 0
            for k in range(n_lsb):
                if bit_idx < need and enc_bits[bit_idx]:
                    v |= (1 << (n_lsb - 1 - k))
                bit_idx += 1
            lsb_vals[i] = v
        vals = (vals & mask) | lsb_vals
        ints[:count] = vals
        slots_used += count
        # write back (share storage)
        new_sd[name] = ints.view(torch.float32).view_as(t)

        if bit_idx >= need:
            break

    stats = {
        "embedded_bits": need,
        "slots_used": slots_used,
        "capacity_bits": cap,
        "n_lsb": n_lsb,
    }
    return new_sd, stats

def extract_bits_from_params(state_dict: Dict[str, torch.Tensor],
                             n_lsb: int = 1,
                             include: Callable[[str, torch.Tensor], bool] = None
                             ) -> Tuple[List[int], dict]:
    """
    Reads the first framed+ECC message by extracting LSBs in the same order.
    We don't know encoded length up front, so we:
      1) Extract a small prefix sufficient to decode the 32-bit header.
      2) Compute total encoded length needed for (header+payload) under Hamming(7,4).
      3) Extract exactly that many bits and return them.
    """
    # Helper: how many Hamming(7,4) bits to carry X payload bits?
    def enc_len_for_payload(payload_len_bits: int) -> int:
        nibbles = (payload_len_bits + 3) // 4
        return nibbles * 7

    # Step 1: grab enough bits to cover header (32 payload bits) after ECC:
    header_enc_len = enc_len_for_payload(_HEADER_BITS)

    # Extract a long enough prefix for safety (header + some body to be robust)
    prefix_need = header_enc_len + enc_len_for_payload(64)  # header + 64 body bits slack
    collected: List[int] = []
    for _, t in iter_param_tensors(state_dict, include):
        flat = t.view(-1).to(torch.float32)
        ints = flat.view(torch.int32)
        if len(collected) >= prefix_need:
            break
        count = min(ints.numel(), (prefix_need - len(collected) + 0) // 1 // 1**0 or prefix_need - len(collected))
        vals = ints[:count]
        for v in vals.tolist():
            for k in range(n_lsb):
                shift = (n_lsb - 1 - k)
                collected.append((v >> shift) & 1)
                if len(collected) >= prefix_need:
                    break
            if len(collected) >= prefix_need:
                break

    if len(collected) < header_enc_len:
        raise ValueError("Not enough bits to recover header prefix.")

    # Decode prefix to learn payload length
    header_dec, _, _ = hamming74_decode_bits(collected[:header_enc_len])
    if len(header_dec) < _HEADER_BITS:
        raise ValueError("Header decode failed.")
    orig_len = bits_to_int(header_dec[:_HEADER_BITS])

    total_enc_needed = header_enc_len + enc_len_for_payload(orig_len)

    # Step 2: extract exactly total_enc_needed bits
    enc_bits: List[int] = []
    for _, t in iter_param_tensors(state_dict, include):
        if len(enc_bits) >= total_enc_needed:
            break
        flat = t.view(-1).to(torch.float32)
        ints = flat.view(torch.int32)
        for v in ints.tolist():
            for k in range(n_lsb):
                shift = (n_lsb - 1 - k)
                enc_bits.append((v >> shift) & 1)
                if len(enc_bits) >= total_enc_needed:
                    break
            if len(enc_bits) >= total_enc_needed:
                break

    if len(enc_bits) < total_enc_needed:
        raise ValueError("Not enough bits to recover full framed message.")

    stats = {
        "encoded_bits_recovered": len(enc_bits),
        "expected_bits": total_enc_needed,
        "n_lsb": n_lsb,
    }
    return enc_bits, stats

# -------------------------
# High-level helpers
# -------------------------
def ecc_encode_data(bitstring: str) -> List[int]:
    """Input: '0101…' from your existing serialize(); Output: ECC-framed bits."""
    return frame_with_len_header(bits_from_str(bitstring))

def ecc_decode_data(encoded_bits: List[int]) -> Tuple[str, dict]:
    """Inverse of ecc_encode_data(). Returns original '0101…' and stats."""
    orig_bits, stats = deframe_and_recover(encoded_bits)
    return bits_to_str(orig_bits), stats

def embed_data_in_parameters(state_dict: Dict[str, torch.Tensor],
                             encoded_bits: List[int],
                             n_lsb: int = 1) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Embed framed+ECC bits (from ecc_encode_data)."""
    return embed_bits_in_params(state_dict, encoded_bits, n_lsb=n_lsb)

def extract_data_from_parameters(state_dict: Dict[str, torch.Tensor],
                                 n_lsb: int = 1) -> Tuple[List[int], dict]:
    """Extract framed+ECC bits, ready for ecc_decode_data()."""
    return extract_bits_from_params(state_dict, n_lsb=n_lsb)

def deserialize_bits_to_items(bitstring: str, deserializer: Callable[[str], object]) -> object:
    """
    Thin wrapper to keep your code clear. Pass your own inverse of `serialize()`.
    Example: deserializer(bitstring) -> original_items
    """
    return deserializer(bitstring)

def verify_equal(original_bits: str, recovered_bits: str) -> bool:
    return original_bits == recovered_bits

def pretty_stats(stats: dict, indent: int = 0) -> str:
    """
    Recursively pretty-print a stats dictionary.
    """
    lines = []
    pad = "  " * indent
    for k, v in stats.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(pretty_stats(v, indent + 1))
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


# -------------------------
# Example usage (pseudo)
# -------------------------
# raw_items = deduplicate(raw_items)
# bitstring = serialize(raw_items)                 # <- your existing function
# enc_bits = ecc_encode_data(bitstring)
# new_state_dict, em_stats = embed_data_in_parameters(model.state_dict(), enc_bits, n_lsb=1)
# save_model(new_state_dict, "model_with_payload.pt")   # <- your existing saver
# enc_bits_rx, ex_stats = extract_data_from_parameters(new_state_dict, n_lsb=1)
# bitstring_rx, ecc_stats = ecc_decode_data(enc_bits_rx)
# ok = verify_equal(bitstring, bitstring_rx)
# recovered_items = deserialize_bits_to_items(bitstring_rx, your_deserialize_fn)


# ===========================
# Distributed-Header + RS Payload (high planes header, interleaved bytes)
# ===========================
# Flexible distributed-header + RS payload for n_lsb in [1..32]
from typing import Dict, List, Tuple, Iterable, Callable, Optional
import torch, zlib, math

# -----------------------
# Configurable defaults
# -----------------------
MAGIC = 0xA5
VERSION = 0x01

# Default RS choices (tuneable)
RS_HEADER_N, RS_HEADER_K = 255, 223   # header codeword (one codeword)
RS_PAY_N, RS_PAY_K = 255, 191         # payload codewords
DEFAULT_INTERLEAVE = 8
# -----------------------
# RS helpers (cached)
# -----------------------
_RS_CACHE = {}
def get_rs(n:int,k:int):
    key=(n,k)
    if key not in _RS_CACHE:
        _RS_CACHE[key] = reedsolo.RSCodec(n-k)
    return _RS_CACHE[key]

def rs_encode(n:int,k:int,data:bytes)->bytes:
    data=(data+bytes(k))[:k]
    return get_rs(n,k).encode(data)

def rs_decode(n: int, k: int, code: bytes) -> tuple[bytes, int]:
    """Decode one RS codeword, return (decoded_k_bytes, num_corrected)."""
    dec, _, errs = get_rs(n, k).decode(code)  # ← no nostrip
    return dec[:k], (0 if errs is None else len(errs))

# -----------------------
# State dict iteration
# -----------------------
def iter_param_tensors(sd:Dict[str,torch.Tensor], include:Optional[Callable]=None):
    for name in sorted(sd.keys()):
        t = sd[name]
        if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
            if include is None or include(name,t):
                yield name,t

def total_params_count(sd:Dict[str,torch.Tensor], include:Optional[Callable]=None)->int:
    return sum(t.numel() for _,t in iter_param_tensors(sd, include))

# -----------------------
# Bit/byte helpers
# -----------------------
# def bytes_to_bits(b:bytes)->List[int]:
#     out=[]
#     for x in b:
#         for k in range(8):
#             out.append((x>>(7-k))&1)
#     return out

# def bits_to_bytes(bits:List[int])->bytes:
#     assert len(bits)%8==0
#     out=bytearray()
#     for i in range(0,len(bits),8):
#         v=0
#         for k in range(8):
#             v=(v<<1)|(1 if bits[i+k] else 0)
#         out.append(v)
#     return bytes(out)

# -----------------------
# Slot planners (flexible for any n_lsb)
# -----------------------
def all_plane_slots(params_count:int, n_lsb:int, planes_order:Optional[List[int]]=None)->List[Tuple[int,int]]:
    """
    Return list of (plane, global_param_idx).
    planes_order: explicit list of plane indices in the order you want them scanned.
    Default: plane-major descending (top planes first).
    """
    if planes_order is None:
        planes_order = list(range(n_lsb-1, -1, -1))  # high -> low
    slots=[]
    for plane in planes_order:
        for p in range(params_count):
            slots.append((plane,p))
    return slots

def pick_header_and_payload_planes(params_count:int, n_lsb:int,
                                   desired_header_planes:int=4,
                                   header_bits_needed:int=None
                                   )->Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """
    Choose header slots (highest planes first) and payload slots (remaining).
    Expand downward if header doesn't fit within desired_header_planes.
    header_bits_needed must be provided (e.g. RS_HEADER_N*8).
    """
    if header_bits_needed is None:
        raise ValueError("header_bits_needed required")

    # start with top `desired_header_planes` planes
    planes = list(range(n_lsb-1, -1, -1))  # high->low
    header_planes = planes[:desired_header_planes]
    # attempt to gather enough header slots, if not enough extend planes downward
    header_slots=[]
    used_plane_set=set()
    for p in header_planes:
        used_plane_set.add(p)
        # collect slots for this plane
        header_slots.extend([(p, idx) for idx in range(params_count)])
        if len(header_slots) >= header_bits_needed:
            break
    # if still short, keep adding next lower plane(s)
    idx=desired_header_planes
    while len(header_slots) < header_bits_needed and idx < len(planes):
        p = planes[idx]
        used_plane_set.add(p)
        header_slots.extend([(p, idx2) for idx2 in range(params_count)])
        idx += 1
    if len(header_slots) < header_bits_needed:
        raise ValueError(f"Not enough total slots for header: need {header_bits_needed}, have {len(header_slots)} (n_lsb={n_lsb}, params={params_count})")

    # finalize header_slots trimmed to needed length
    header_slots = header_slots[:header_bits_needed]

    # payload planes = all planes except used_plane_set, keep same high->low order
    payload_planes = [p for p in planes if p not in used_plane_set]
    payload_slots = []
    for p in payload_planes:
        payload_slots.extend([(p, idx) for idx in range(params_count)])
    return header_slots, payload_slots

# -----------------------
# Bit IO into chosen slots
# -----------------------
def set_bits_into_slots(sd:Dict[str,torch.Tensor],
                        slot_list:Iterable[Tuple[int,int]],
                        bits_iter:Iterable[int],
                        include:Optional[Callable]=None):
    """Write bits in order to specified (plane,global_idx) slots. Stops early if bits_iter exhausted."""
    it = iter(bits_iter)
    pending = {}
    for pos,(plane,gidx) in enumerate(slot_list):
        pending.setdefault(gidx, []).append((plane,pos))
    # convert pending to dict to mutate
    pending = {k:v[:] for k,v in pending.items()}
    g=0
    for _,t in iter_param_tensors(sd, include):
        flat = t.view(-1).to(torch.float32)
        ints = flat.view(torch.int32)
        for i in range(ints.numel()):
            if g in pending:
                v = ints[i].item()
                # for each requested plane at this global param index
                for plane,pos in pending[g]:
                    b = next(it, None)
                    if b is None:
                        break
                    mask = ~(1 << plane)
                    v = (v & mask) | ((1 if b else 0) << plane)
                ints[i] = v
                del pending[g]
            g += 1
            if not pending:
                break
        # write-back
        t.copy_(ints.view(torch.float32).view_as(t))
        if not pending:
            break
    # we don't return remaining bits; caller should ensure slot_list len >= bits count

def get_bits_from_slots(sd:Dict[str,torch.Tensor],
                        slot_list:Iterable[Tuple[int,int]],
                        include:Optional[Callable]=None)->List[int]:
    want = list(slot_list)
    out = [0]*len(want)
    buckets={}
    for pos,(plane,gidx) in enumerate(want):
        buckets.setdefault(gidx, []).append((plane,pos))
    g=0
    for _,t in iter_param_tensors(sd, include):
        flat = t.view(-1).to(torch.float32)
        ints = flat.view(torch.int32)
        for i in range(ints.numel()):
            if g in buckets:
                v = ints[i].item()
                for plane,pos in buckets[g]:
                    out[pos] = (v >> plane) & 1
                del buckets[g]
            g += 1
            if not buckets:
                return out
    return out

# -----------------------
# Header pack/unpack (same as before)
# -----------------------
def build_header_bytes(orig_len_bytes:int,total_payload_codewords:int, rs_n:int, rs_k:int, interleave_d:int)->bytes:
    core = bytearray()
    core += bytes([MAGIC, VERSION, rs_n & 0xFF, rs_k & 0xFF, interleave_d & 0xFF, 0x00])
    core += orig_len_bytes.to_bytes(4,"big")
    core += total_payload_codewords.to_bytes(4,"big")
    core += b"\x00\x00\x00\x00"
    hcrc = zlib.crc32(core) & 0xFFFF
    core += hcrc.to_bytes(2,"big")
    return bytes(core)

def parse_header_bytes(kdata:bytes)->dict:
    if len(kdata) < 20:
        raise ValueError("Header data too short")
    magic,ver,rsn,rsk,idepth,_ = kdata[0:6]
    orig_len = int.from_bytes(kdata[6:10],"big")
    codewords = int.from_bytes(kdata[10:14],"big")
    got_crc = int.from_bytes(kdata[18:20],"big")
    want_crc = zlib.crc32(kdata[:18]) & 0xFFFF
    if magic!=MAGIC or ver!=VERSION or got_crc!=want_crc:
        raise ValueError("Header CRC/version/magic check failed")
    return {"orig_len_bytes":orig_len,"rs_n":rsn,"rs_k":rsk,"interleave_d":idepth,"total_payload_codewords":codewords}

# -----------------------
# Payload RS encode/decode (with interleave)
# -----------------------
def encode_payload_rs(payload:bytes, rs_n:int, rs_k:int, interleave_d:int)->bytes:
    payload = payload + zlib.crc32(payload).to_bytes(4,"big")
    chunks = [payload[i:i+rs_k] for i in range(0,len(payload),rs_k)]
    if len(chunks)==0:
        chunks=[b""]
    chunks[-1] = (chunks[-1] + bytes(rs_k))[:rs_k]
    streams = [bytearray() for _ in range(interleave_d)]
    for i,ch in enumerate(chunks):
        streams[i % interleave_d] += rs_encode(rs_n, rs_k, ch)
    return b"".join(streams)

def decode_payload_rs(codewords_concat: bytes, rs_n: int, rs_k: int,
                      interleave_d: int, expected_len: int, total_chunks: int) -> tuple[bytes, dict]:
    cw_len = rs_n
    assert len(codewords_concat) % cw_len == 0, "payload bytes not a multiple of RS_n"
    total_cw = len(codewords_concat) // cw_len
    if total_cw != total_chunks:
        # Safety net: trust the header count, not raw length
        total_cw = total_chunks

    # Split into D streams with uneven lengths:
    q, r = divmod(total_cw, interleave_d)  # first r streams have (q+1) codewords
    streams = []
    off = 0
    for s in range(interleave_d):
        n_cw = q + (1 if s < r else 0)
        bytes_len = n_cw * cw_len
        streams.append(codewords_concat[off:off+bytes_len])
        off += bytes_len

    # Deinterleave back to original chunk order:
    chunks = []
    corr_total = 0
    for i in range(q + 1):  # up to q or q+1 for the first r streams
        for s in range(interleave_d):
            n_cw_s = q + (1 if s < r else 0)
            if i >= n_cw_s:
                continue
            cw = streams[s][i*cw_len:(i+1)*cw_len]
            kdata, corr = rs_decode(rs_n, rs_k, cw)
            chunks.append(kdata)
            corr_total += corr

    data = b"".join(chunks)
    data = data[:expected_len + 4]  # payload + CRC32
    payload, crc = data[:-4], int.from_bytes(data[-4:], "big")
    ok = (zlib.crc32(payload) & 0xFFFFFFFF) == crc
    return payload, {"corrected_symbols": corr_total, "crc_ok": ok}


# -----------------------
# Top-level flexible embed/extract
# -----------------------
def embed_flexible(sd:Dict[str,torch.Tensor],
                   bitstring:str,
                   *,
                   n_lsb:int=20,
                   desired_header_planes:int=4,
                   rs_header_n:int=RS_HEADER_N, rs_header_k:int=RS_HEADER_K,
                   rs_pay_n:int=RS_PAY_N, rs_pay_k:int=RS_PAY_K,
                   interleave_d:int=DEFAULT_INTERLEAVE,
                   include:Optional[Callable]=None)->Tuple[Dict[str,torch.Tensor],dict]:
    """
    Fits header into top planes (expand if needed), payload into remaining planes.
    Works for n_lsb in [1,32] (or higher if you change n_lsb input).
    """
    assert 1 <= n_lsb <= 32, "n_lsb supported in [1..32]"
    params_count = total_params_count(sd, include)

    # payload as bytes (keep leading zeros)
    payload = int(bitstring,2).to_bytes((len(bitstring)+7)//8, "big") if bitstring else b""
    # compute chunks & codewords counts
    total_chunks = math.ceil((len(payload)+4)/rs_pay_k)
    #total_cw_rounded = math.ceil(total_chunks / interleave_d) * interleave_d

    # header bytes (core small; will be RS(HEADER_N,HEADER_K) encoded)
    header_core = build_header_bytes(len(payload), total_chunks, rs_pay_n, rs_pay_k, interleave_d)    
    header_cw = rs_encode(rs_header_n, rs_header_k, header_core)  # length rs_header_n bytes
    header_bits_needed = len(header_cw) * 8

    # pick header and payload slots
    header_slots, payload_slots = pick_header_and_payload_planes(params_count, n_lsb,
                                                                 desired_header_planes=desired_header_planes,
                                                                 header_bits_needed=header_bits_needed)
    # encode payload
    payload_cw_concat = encode_payload_rs(payload, rs_pay_n, rs_pay_k, interleave_d)
    payload_bits = bytes_to_bits(payload_cw_concat)

    if len(payload_bits) > len(payload_slots):
        raise ValueError(f"Not enough payload capacity: need {len(payload_bits)} bits, have {len(payload_slots)} bits (n_lsb={n_lsb}, params={params_count})")

    # write into copies of the state dict
    new_sd = {k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in sd.items()}

    # write header bits (spread 1 bit per slot)
    header_bits = bytes_to_bits(header_cw)
    set_bits_into_slots(new_sd, header_slots, header_bits, include=include)

    # write payload bits into payload_slots
    set_bits_into_slots(new_sd, payload_slots[:len(payload_bits)], payload_bits, include=include)

    stats = {"params":params_count,
             "n_lsb":n_lsb,
             "header_bits_written":len(header_bits),
             "payload_bits_written":len(payload_bits),
             "payload_bytes_raw":len(payload),
             "rs_payload": {"n":rs_pay_n,"k":rs_pay_k,"interleave_d":interleave_d,"total_chunks":total_chunks}}
    return new_sd, stats

def extract_flexible(sd:Dict[str,torch.Tensor],
                     *,
                     n_lsb:int=20,
                     desired_header_planes:int=4,
                     rs_header_n:int=RS_HEADER_N, rs_header_k:int=RS_HEADER_K,
                     include:Optional[Callable]=None)->Tuple[str,dict]:
    """
    Reverse of embed_flexible. Reads header from top planes (as embedded),
    then reads payload slots accordingly and RS-decodes.
    """
    params_count = total_params_count(sd, include)
    # header_bits_needed
    header_bits_needed = rs_header_n * 8
    header_slots, payload_slots = pick_header_and_payload_planes(params_count, n_lsb,
                                                                 desired_header_planes=desired_header_planes,
                                                                 header_bits_needed=header_bits_needed)
    header_bits = get_bits_from_slots(sd, header_slots, include=include)
    header_bytes = bits_to_bytes(header_bits)
    kdata, corr = rs_decode(rs_header_n, rs_header_k, header_bytes)
    hdr = parse_header_bytes(kdata)
    # compute payload bytes to read
    total_cw = hdr["total_payload_codewords"]
    need_bytes = total_cw * hdr["rs_n"]
    need_bits = need_bytes * 8
    if need_bits > len(payload_slots):
        raise ValueError(f"Declared payload needs {need_bits} payload bits but only {len(payload_slots)} available (n_lsb={n_lsb})")
    p_bits = get_bits_from_slots(sd, payload_slots[:need_bits], include=include)
    p_bytes = bits_to_bytes(p_bits)
    # RS decode payload

    payload, ecc_stats = decode_payload_rs(
        p_bytes, hdr["rs_n"], hdr["rs_k"], hdr["interleave_d"],
        expected_len=hdr["orig_len_bytes"],
        total_chunks=total_cw
    )   
     # produce bitstring of exact length
    orig_bits_len = hdr["orig_len_bytes"] * 8
    if len(payload)==0:
        bits_str = ""
    else:
        bits_str = bin(int.from_bytes(payload,"big"))[2:].zfill(orig_bits_len)
    stats = {"header": {"corrected_symbols":corr, **hdr}, "payload":ecc_stats, "n_lsb":n_lsb}
    return bits_str, stats


#=========================
# only duplication and vertical encoding
#=========================

import numpy as np
from typing import Tuple

# ---------- helpers ----------
def _bits_str_to_list(bitstr: str):
    return [1 if c == "1" else 0 for c in bitstr]

def _bits_list_to_str(bits):
    return "".join("1" if b else "0" for b in bits)

# ---------- 2) vertical embed (MSB→LSB across bytes, left→right across the byte stream) ----------
def embed_vertical_from_bitstring(params: np.ndarray, bitstream: str, n: int) -> Tuple[np.ndarray, int]:
    """
    Vertical embedding:
      for bit_pos in 7..0:
        for byte_idx in 0..N-1:
           write next payload bit into that bit position of that byte
    Payload = bitstream duplicated n times (concatenation).
    Returns (modified_params, bits_written).
    NOTE: operates on ALL bits of the float32 bytes (sign/exponent/mantissa).
    """
    if params.dtype != np.float32:
        raise ValueError("params must be float32")
    p = params.copy()
    byteview = p.view(np.uint8)
    capacity_bits = byteview.size * 8

    payload_bits = _bits_str_to_list(bitstream) * n
    bits_to_write = min(len(payload_bits), capacity_bits)

    k = 0
    for bit_pos in range(7, -1, -1):
        for bi in range(byteview.size):
            if k >= bits_to_write:
                break
            b = int(byteview[bi])
            bit = payload_bits[k]
            byteview[bi] = ((b & ~(1 << bit_pos)) | (bit << bit_pos)) & 0xFF
            k += 1
        if k >= bits_to_write:
            break
    return p, k

# ---------- 3) capacity + duplication factor n ----------
import torch
import numpy as np

def compute_model_capacity_bits(model_weights) -> int:
    """
    Compute total vertical embedding capacity in bits.
    Each float32 parameter contributes 32 bits × number of elements.
    """
    total_elems = 0
    for v in model_weights.values():
        if not torch.is_floating_point(v):
            continue  # skip non-float tensors (like buffers or ints)
        total_elems += v.numel()
    return total_elems * 32

def choose_multiplier_for_capacity(payload_bits: int, capacity_bits: int) -> Tuple[int, bool]:
    """
    Largest n so n * payload_bits <= capacity_bits.
    If payload_bits > capacity_bits -> (1, True) meaning n=1 and we will truncate at capacity.
    """
    if payload_bits == 0:
        return 0, False
    if payload_bits > capacity_bits:
        return 1, True
    return max(1, capacity_bits // payload_bits), False

# ---------- 4) vertical decode (inverse read order) ----------
def decode_vertical_to_bitstring(params: np.ndarray, expected_bits: int = None) -> str:
    """
    Reads bits in the same vertical order (MSB→LSB across bytes) and returns a '0'/'1' string.
    If expected_bits is provided, stops after that many bits (use bits_written from embed).
    """
    if params.dtype != np.float32:
        raise ValueError("params must be float32")
    byteview = params.view(np.uint8)
    capacity_bits = byteview.size * 8
    bits_to_read = capacity_bits if expected_bits is None else min(expected_bits, capacity_bits)

    out = []
    r = 0
    for bit_pos in range(7, -1, -1):
        for bi in range(byteview.size):
            if r >= bits_to_read:
                break
            out.append((int(byteview[bi]) >> bit_pos) & 1)
            r += 1
        if r >= bits_to_read:
            break
    return _bits_list_to_str(out)
