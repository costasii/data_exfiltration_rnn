import random
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


def deinterleave_bits(bits: str, cols: int, pad_bits_added: int) -> str:
    """
    Inverse of interleave_bits.
    """
    if cols <= 1 or not bits:
        return bits

    rows = len(bits) // cols
    out = ["0"] * (rows * cols)

    idx = 0
    for c in range(cols):
        for r in range(rows):
            out[r * cols + c] = bits[idx]
            idx += 1

    res = "".join(out)
    if pad_bits_added:
        res = res[:-pad_bits_added]
    return res


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


def _rand_bits(n): 
    return "".join(random.choice("01") for _ in range(n))

for _ in range(1000):
    plain = _rand_bits(11 * 20 + random.randint(0, 10))  # random length
    coded, enc_dbg = hamming15_11_encode_bits(plain)
    rec, dec_dbg = hamming15_11_decode_bits(coded, plain_bits_len=len(plain))
    assert rec == plain, (enc_dbg, dec_dbg)

print("Hamming(15,11) encode/decode identity: OK")
