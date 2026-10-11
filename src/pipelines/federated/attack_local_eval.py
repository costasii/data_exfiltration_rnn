from __future__ import annotations

from collections import Counter
from typing import Sequence, Optional, Tuple, Dict, List

import torch

import traceback
from pathlib import Path
import json


def split_by_boundary_pair(
    stream: Sequence[int],
    esc_event: int,
    delim_event: int,
) -> Tuple[List[List[int]], int]:
    """
    Split a decoded/target stream into sequences using duplicated boundary markers:
      [esc, delim, esc, delim]

    Boundary tokens are removed from returned sequences.
    Returns:
      sequences, num_boundaries_found
    """
    pattern = [int(esc_event), int(delim_event), int(esc_event), int(delim_event)]
    seqs: List[List[int]] = []
    cur: List[int] = []
    i = 0
    boundaries = 0

    stream_int = [int(x) for x in stream]

    while i < len(stream_int):
        if i <= len(stream_int) - 4 and stream_int[i:i + 4] == pattern:
            seqs.append(cur)
            cur = []
            boundaries += 1
            i += 4
        else:
            cur.append(stream_int[i])
            i += 1

    seqs.append(cur)
    return seqs, boundaries

def compute_macro_f1_event_similarity(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    ignore_labels: set[int],
) -> Dict[str, float]:
    """
    Macro-F1 over encoded event classes, where every wrong prediction
    is mapped to OTHER, regardless of whether it is in-vocab or OOV.
    """
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]

    pairs = []
    for t, p in zip(y_true, y_pred):
        if t in ignore_labels:
            continue
        pairs.append((t, p))

    OTHER = "__OTHER__"

    encoded_event_labels = sorted({t for t, _ in pairs})
    eval_classes = encoded_event_labels + [OTHER]

    tp = {c: 0 for c in eval_classes}
    fp = {c: 0 for c in eval_classes}
    fn = {c: 0 for c in eval_classes}

    for t, p in pairs:
        true_c = t
        pred_c = t if p == t else OTHER

        if pred_c == true_c:
            tp[true_c] += 1
        else:
            fn[true_c] += 1
            fp[pred_c] += 1

    f1s = []
    for c in eval_classes:
        prec_d = tp[c] + fp[c]
        rec_d = tp[c] + fn[c]
        prec = tp[c] / prec_d if prec_d > 0 else 0.0
        rec = tp[c] / rec_d if rec_d > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "encoded_num_classes": float(len(encoded_event_labels)),
        "f1_macro_event_similarity": float(macro_f1),
    }


def evaluate_trigger_attack_from_model(
    model: torch.nn.Module,
    trigger_patterns: Sequence[Sequence[int]],
    target_labels: Sequence[int],
    *,
    device: str | torch.device,
    window_size: int,
    input_size: int,
    num_classes: int,
    subtract_1: bool,
    esc_event: int,
    delim_event: int,
    total_train_events: Optional[int] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Evaluate trigger-set attack directly from an in-memory model.

    Returns:
      decoded_stream, metrics
    """
    if len(trigger_patterns) != len(target_labels):
        raise ValueError(
            f"trigger_patterns ({len(trigger_patterns)}) must match "
            f"target_labels ({len(target_labels)})"
        )

    model = model.to(device)
    model.eval()

    decoded_stream: List[int] = []

    for idx, trigger in enumerate(trigger_patterns):
        try:
            if subtract_1:
                trig0 = [int(e) - 1 for e in trigger]
            else:
                trig0 = [int(e) for e in trigger]

            seq0 = torch.tensor(trig0, dtype=torch.float).view(
                1, window_size, input_size
            ).to(device)

            seq1 = [0] * num_classes
            cnt = Counter(trig0)

            for k, v in cnt.items():
                if 0 <= k < num_classes:
                    seq1[k] = v

            seq1 = torch.tensor(seq1, dtype=torch.float).view(
                1, num_classes, input_size
            ).to(device)

            with torch.no_grad():
                out = model(features=[seq0, seq1], device=device)
                pred0 = int(torch.argmax(out, dim=1).item())

            decoded_stream.append(pred0)

        except Exception as e:

            error_payload = {
                "method": "evaluate_trigger_attack_from_model",
                "trigger_index": idx,
                "trigger_len": len(trigger),
                "window_size": window_size,
                "input_size": input_size,
                "num_classes": num_classes,
                "subtract_1": subtract_1,
                "target_labels_len": len(target_labels),
                "trigger_patterns_len": len(trigger_patterns),
                "trigger_head": list(trigger)[:50],
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "traceback": traceback.format_exc(),
            }

            debug_path = Path("debug_trigger_eval_error.json")

            with debug_path.open("w", encoding="utf-8") as f:
                json.dump(error_payload, f, indent=2)

            raise

    metrics: Dict[str, float] = {}
    num_unique = len(trigger_patterns)

    # Trigger accuracy
    correct_triggers = sum(
        int(int(t) == int(p)) for t, p in zip(target_labels, decoded_stream)
    )
    metrics["trigger_correct"] = float(correct_triggers)
    metrics["trigger_total"] = float(len(target_labels))
    metrics["trigger_accuracy"] = (
        correct_triggers / len(target_labels) if len(target_labels) > 0 else 0.0
    )

    # Delimiter token accuracy
    delim_positions = [
        i for i, t in enumerate(target_labels)
        if int(t) in (int(esc_event), int(delim_event))
    ]
    delim_total = len(delim_positions)
    delim_correct = sum(
        1 for i in delim_positions
        if int(decoded_stream[i]) == int(target_labels[i])
    )

    metrics["delimiter_token_correct"] = float(delim_correct)
    metrics["delimiter_token_total"] = float(delim_total)
    metrics["delimiter_token_accuracy"] = (
        delim_correct / delim_total if delim_total > 0 else 0.0
    )

    # Boundary-level delimiter accuracy
    pattern = [int(esc_event), int(delim_event), int(esc_event), int(delim_event)]
    boundary_starts = []
    for i in range(len(target_labels) - 3):
        if [int(x) for x in target_labels[i:i + 4]] == pattern:
            boundary_starts.append(i)

    boundary_total = len(boundary_starts)
    boundary_recognized = 0
    boundary_pair1_ok = 0
    boundary_pair2_ok = 0
    boundary_tokens_correct_sum = 0

    for i in boundary_starts:
        d4 = [int(x) for x in decoded_stream[i:i + 4]]

        pair1 = (d4[0] == esc_event and d4[1] == delim_event)
        pair2 = (d4[2] == esc_event and d4[3] == delim_event)

        if pair1:
            boundary_pair1_ok += 1
        if pair2:
            boundary_pair2_ok += 1
        if pair1 or pair2:
            boundary_recognized += 1

        boundary_tokens_correct_sum += sum(int(d4[j] == pattern[j]) for j in range(4))

    metrics["boundary_total"] = float(boundary_total)
    metrics["boundary_recognized"] = float(boundary_recognized)
    metrics["boundary_accuracy"] = (
        boundary_recognized / boundary_total if boundary_total > 0 else 0.0
    )
    metrics["boundary_pair1_accuracy"] = (
        boundary_pair1_ok / boundary_total if boundary_total > 0 else 0.0
    )
    metrics["boundary_pair2_accuracy"] = (
        boundary_pair2_ok / boundary_total if boundary_total > 0 else 0.0
    )
    metrics["boundary_avg_token_accuracy"] = (
        boundary_tokens_correct_sum / (4 * boundary_total) if boundary_total > 0 else 0.0
    )

    # General sizes
    metrics["num_unique_triggers"] = float(num_unique)
    metrics["encoded_stream_len"] = float(len(target_labels))
    metrics["decoded_stream_len"] = float(len(decoded_stream))

    # Event accuracy, excluding delimiter targets
    event_total = 0
    event_correct = 0
    for t, p in zip(target_labels, decoded_stream):
        t = int(t)
        p = int(p)
        if t == int(esc_event) or t == int(delim_event):
            continue
        event_total += 1
        if p == t:
            event_correct += 1

    metrics["event_correct"] = float(event_correct)
    metrics["event_total"] = float(event_total)
    metrics["event_accuracy"] = (
        event_correct / event_total if event_total > 0 else 0.0
    )

    # Event membership
    target_set = {
        int(t) for t in target_labels
        if int(t) not in (int(esc_event), int(delim_event))
    }
    decoded_set = {
        int(p) for p in decoded_stream
        if int(p) not in (int(esc_event), int(delim_event))
    }
    denom = len(target_set) if len(target_set) > 0 else 1

    metrics["encoded_unique_event_count"] = float(len(target_set))
    metrics["decoded_unique_event_count"] = float(len(decoded_set))
    metrics["event_membership"] = len(decoded_set & target_set) / denom

    if total_train_events is not None:
        metrics["total_train_events"] = float(total_train_events)

    # Sequence reconstruction
    target_seqs, target_boundaries = split_by_boundary_pair(
        target_labels, esc_event, delim_event
    )
    decoded_seqs, decoded_boundaries = split_by_boundary_pair(
        decoded_stream, esc_event, delim_event
    )

    metrics["target_boundaries_found"] = float(target_boundaries)
    metrics["decoded_boundaries_found"] = float(decoded_boundaries)
    metrics["target_num_sequences"] = float(len(target_seqs))
    metrics["decoded_num_sequences"] = float(len(decoded_seqs))

    seq_acc_list: List[float] = []
    fully_encoded_count = 0

    segmentation_ok = (
        decoded_boundaries == target_boundaries
        and len(decoded_seqs) == len(target_seqs)
    )

    if segmentation_ok:
        for t_seq, d_seq in zip(target_seqs, decoded_seqs):
            if len(d_seq) != len(t_seq) or len(t_seq) == 0:
                continue
            correct = sum(int(p == t) for p, t in zip(d_seq, t_seq))
            seq_acc_list.append(correct / len(t_seq))
            fully_encoded_count += 1

    metrics["fully_encoded_sequences"] = float(fully_encoded_count)
    metrics["mean_sequence_accuracy"] = (
        float(sum(seq_acc_list) / fully_encoded_count)
        if fully_encoded_count > 0 else 0.0
    )

    seq_total = len(target_seqs) if len(target_seqs) > 0 else 1
    m = min(len(target_seqs), len(decoded_seqs))
    seq_correct = 0
    for i in range(m):
        if decoded_seqs[i] == target_seqs[i]:
            seq_correct += 1

    metrics["sequence_correct"] = float(seq_correct)
    metrics["sequence_total"] = float(len(target_seqs))
    metrics["sequence_accuracy"] = seq_correct / seq_total


    f1_metrics = compute_macro_f1_event_similarity(
        y_true=target_labels,
        y_pred=decoded_stream,
        ignore_labels={int(esc_event), int(delim_event), -1},
        )
    metrics.update(f1_metrics)

    return decoded_stream, metrics