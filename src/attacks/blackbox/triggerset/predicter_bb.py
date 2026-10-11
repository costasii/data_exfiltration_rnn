#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter
sys.path.append('../../')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List, Tuple
from collections import Counter
import torch


from src.data.log import log_dataset


from src.attacks.blackbox.triggerset.bb_utils import log_confusion_matrix_restricted, split_by_boundary_pair, compute_macro_f1_event_similarity

def generate(data_dir, file_name, window_size, *, subtract_1: bool = True):
    sessions = {}
    length = 0
    file_path = Path(data_dir) / file_name

    with open(file_path, "r", encoding="utf-8") as f:
        for ln in f.readlines():
            ln = ln.strip()
            if not ln:
                continue

            if subtract_1:
                seq = list(map(lambda n: n - 1, map(int, ln.split())))
            else:
                seq = list(map(int, ln.split()))

            seq = seq + [-1] * (window_size + 1 - len(seq))
            t = tuple(seq)
            sessions[t] = sessions.get(t, 0) + 1
            length += 1

    print(f"Number of sessions({file_name}): {len(sessions)}")
    return sessions, length

class Predicter():
    def __init__(self, model, options, *, dataset_dir, dataset_files, subtract_1: bool = True):
        self.data_dir = Path(dataset_dir)
        self.dataset_files = dataset_files
        self.subtract_1 = subtract_1

        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']

        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']   # <-- use this instead of 33
        self.input_size = options['input_size']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']

        self.test_normal_loader, self.test_normal_length = generate(
            self.data_dir,
            self.dataset_files['test_normal'],
            self.window_size,
            subtract_1=self.subtract_1,
        )
        self.test_abnormal_loader, self.test_abnormal_length = generate(
            self.data_dir,
            self.dataset_files['test_abnormal'],
            self.window_size,
            subtract_1=self.subtract_1,
        )


    def predict_unsupervised(self, return_metrics=False):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print(f'model_path: {self.model_path}')
       
        TP, FP = 0, 0
        start_time = time.time()

        # Test the model
        with torch.no_grad():
            for line in tqdm(self.test_normal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] *  self.num_classes
                    log_counter = Counter(seq0)
                    for key, cnt in log_counter.items():
                        if 0 <= key < self.num_classes:
                            seq1[key] = cnt
                    seq0 = torch.tensor(seq0, dtype=torch.float).view(-1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(-1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)

                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]

                    if label not in predicted:
                        FP += self.test_normal_loader[line]
                        break
            for line in tqdm(self.test_abnormal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * self.num_classes
                    log_counter = Counter(seq0)
                    for key, cnt in log_counter.items():
                        if 0 <= key < self.num_classes:
                            seq1[key] = cnt


                    seq0 = torch.tensor(seq0, dtype=torch.float).view(-1, self.window_size, self.input_size).to(self.device)
                    seq1 = torch.tensor(seq1, dtype=torch.float).view(-1, self.num_classes, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)

                    output = model(features=[seq0, seq1], device=self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]

                    if label not in predicted:
                        TP += self.test_abnormal_loader[line]
                        break

        # Compute precision, recall, F1
        FN = self.test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        elapsed_time = time.time() - start_time

        print(f"False positive (FP): {FP}, False negative (FN): {FN}, Precision: {P:.3f}%, Recall: {R:.3f}%, F1: {F1:.3f}%")
        print(f"Elapsed time: {elapsed_time}")
        
        if return_metrics:
            return P, R, F1, FP, FN, elapsed_time
            


    def evaluate_trigger_attack(
        self,
        trigger_patterns: Sequence[Sequence[int]],
        target_labels: Sequence[int],                 # encoded event stream (targets from the training set)
        log_to_wandb: bool = True,
        total_train_events: Optional[int] = None,     # for attack_capacity
        orig_encoded_seqs: Optional[Sequence[Sequence[int]]] = None,
        subseq_len: Optional[int] = None,
        esc_event: int = 0,                               
        delim_event: int = 0,    
        # confusion matrix options
        log_confusion_matrix: bool = True,
        include_predicted_labels_in_cm: bool = True,
        cm_wandb_key: str = "trigger_confusion_matrix_restricted",

        # optional preview table
        log_decoded_table: bool = True,
        decoded_table_limit: int = 200,
 

    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Decode via triggers -> compute metrics from decoded stream -> (optionally) log confusion matrix.
        Returns (decoded_stream, metrics).
        """

        if len(trigger_patterns) != len(target_labels):
            raise ValueError(
                f"trigger_patterns ({len(trigger_patterns)}) must match target_labels ({len(target_labels)})"
            )
        num_unique = len(trigger_patterns)

        # --- load model once ---
        ckpt = torch.load(self.model_path, map_location=self.device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # --- simluate querying the model ---
        decoded_stream: List[int] = []

        for trigger in trigger_patterns:
            if self.subtract_1:
                trig0 = [int(e) - 1 for e in trigger]
            else:
                trig0 = [int(e) for e in trigger]
            

            seq0 = torch.tensor(trig0, dtype=torch.float).view(
                1, self.window_size, self.input_size
            ).to(self.device)

            seq1 = [0] * self.num_classes
            cnt = Counter(trig0)
            for k, v in cnt.items():
                if 0 <= k < self.num_classes:
                    seq1[k] = v
            seq1 = torch.tensor(seq1, dtype=torch.float).view(
                1, self.num_classes, self.input_size
            ).to(self.device)

            with torch.no_grad():
                out = self.model(features=[seq0, seq1], device=self.device)  # [1, C]
                pred0 = int(torch.argmax(out, dim=1).item())                 

            decoded_stream.append(pred0)  

        print(target_labels)
        print(decoded_stream)

        delim_positions = [i for i, t in enumerate(target_labels) if int(t) in (esc_event, delim_event)]
        delim_total = len(delim_positions)
        delim_correct = sum(
            1 for i in delim_positions
            if int(decoded_stream[i]) == int(target_labels[i])
        )

        # boundary detection
        # --- Boundary-level delimiter accuracy for duplicated markers [esc,delim,esc,delim] ---
        pattern = [int(esc_event), int(delim_event), int(esc_event), int(delim_event)]
        boundary_starts = []

        # find exact boundary locations in the *target* stream
        for i in range(len(target_labels) - 3):
            if [int(x) for x in target_labels[i:i+4]] == pattern:
                boundary_starts.append(i)

        boundary_total = len(boundary_starts)
        boundary_recognized = 0
        boundary_pair1_ok = 0
        boundary_pair2_ok = 0
        boundary_tokens_correct_sum = 0  # optional diagnostic

        for i in boundary_starts:
            d4 = [int(x) for x in decoded_stream[i:i+4]]

            pair1 = (d4[0] == esc_event and d4[1] == delim_event)
            pair2 = (d4[2] == esc_event and d4[3] == delim_event)

            if pair1:
                boundary_pair1_ok += 1
            if pair2:
                boundary_pair2_ok += 1
            if pair1 or pair2:
                boundary_recognized += 1

            # optional: how many of the 4 tokens are correct at this boundary block
            boundary_tokens_correct_sum += sum(int(d4[j] == pattern[j]) for j in range(4))


     
        metrics: Dict[str, float] = {}

        # --- NEW: per-trigger results + trigger accuracy (ALL triggers, incl delimiters) ---
        preds_list = []
        correct_triggers = 0

        for i, (trigger, target, pred) in enumerate(zip(trigger_patterns, target_labels, decoded_stream)):
            target = int(target)
            pred = int(pred)
            is_correct = int(pred == target)
            correct_triggers += is_correct

            preds_list.append({
                "unique_index": i,
                "trigger": list(map(int, trigger)),
                "target": target,
                "predicted_top1": pred,
                "correct": bool(is_correct),
            })

        metrics["trigger_correct"] = float(correct_triggers)
        metrics["trigger_total"] = float(len(target_labels) if len(target_labels) > 0 else 0)
        metrics["trigger_accuracy"] = (correct_triggers / len(target_labels)) if len(target_labels) > 0 else 0.0

        metrics["delimiter_token_correct"] = float(delim_correct)
        metrics["delimiter_token_total"] = float(delim_total)
        metrics["delimiter_token_accuracy"] = (delim_correct / delim_total) if delim_total > 0 else 0.0

        metrics["boundary_total"] = float(boundary_total)
        metrics["boundary_recognized"] = float(boundary_recognized)
        metrics["boundary_accuracy"] = (boundary_recognized / boundary_total) if boundary_total > 0 else 0.0

        # optional diagnostics (super useful)
        metrics["boundary_pair1_accuracy"] = (boundary_pair1_ok / boundary_total) if boundary_total > 0 else 0.0
        metrics["boundary_pair2_accuracy"] = (boundary_pair2_ok / boundary_total) if boundary_total > 0 else 0.0
        metrics["boundary_avg_token_accuracy"] = (boundary_tokens_correct_sum / (4 * boundary_total)) if boundary_total > 0 else 0.0



       
        # --- metrics (all based on decoded stream) ---
        metrics["num_unique_triggers"] = float(num_unique)
        metrics["encoded_stream_len"] = float(len(target_labels))
        metrics["decoded_stream_len"] = float(len(decoded_stream))

        # Event accuracy (DATA EVENTS ONLY) computed on original alignment:
        # skip positions where the TARGET is a delimiter token
        event_total = 0
        event_correct = 0

        for t, p in zip(target_labels, decoded_stream):
            t = int(t)
            p = int(p)

            # skip delimiter targets (do not count them as events)
            if t == int(esc_event) or t == int(delim_event):
                continue

            event_total += 1
            if p == t:
                event_correct += 1

        metrics["event_correct"] = float(event_correct)
        metrics["event_total"] = float(event_total if event_total > 0 else 0)
        metrics["event_accuracy"] = (event_correct / event_total) if event_total > 0 else 0.0

        # Event membership over DATA EVENTS ONLY (exclude delimiter tokens from sets)
        target_set = set(int(t) for t in target_labels if int(t) not in (int(esc_event), int(delim_event)))
        decoded_set = set(int(p) for p in decoded_stream if int(p) not in (int(esc_event), int(delim_event)))

        denom = len(target_set) if len(target_set) > 0 else 1
        metrics["encoded_unique_event_count"] = float(len(target_set))
        metrics["decoded_unique_event_count"] = float(len(decoded_set))
        metrics["event_membership"] = len(decoded_set & target_set) / denom


        if total_train_events is not None:
            metrics["total_train_events"] = float(total_train_events)
        
        # sequence coverage
        target_seqs, target_boundaries = split_by_boundary_pair(target_labels, esc_event, delim_event)
        decoded_seqs, decoded_boundaries = split_by_boundary_pair(decoded_stream, esc_event, delim_event)


        metrics["target_boundaries_found"] = float(target_boundaries)
        metrics["decoded_boundaries_found"] = float(decoded_boundaries)
        metrics["target_num_sequences"] = float(len(target_seqs))
        metrics["decoded_num_sequences"] = float(len(decoded_seqs))


        # --- NEW: per-sequence event accuracy (ONLY fully-encoded sequences) ---
        # Here, split_by_boundary_pair(...) should already REMOVE boundary tokens,
        # so delimiters are automatically not counted.

        seq_acc_list: List[float] = []
        fully_encoded_count = 0

        # "Fully encoded" = segmentation matches (same #boundaries / seqs) AND this seq has full length
        segmentation_ok = (decoded_boundaries == target_boundaries) and (len(decoded_seqs) == len(target_seqs))

        if segmentation_ok:
            for t_seq, d_seq in zip(target_seqs, decoded_seqs):
                # if a sequence is not fully encoded, skip it
                if len(d_seq) != len(t_seq) or len(t_seq) == 0:
                    continue

                correct = sum(int(p == t) for p, t in zip(d_seq, t_seq))
                acc = correct / len(t_seq)

                seq_acc_list.append(acc)
                fully_encoded_count += 1

        metrics["fully_encoded_sequences"] = float(fully_encoded_count)
        metrics["mean_sequence_accuracy"] = (
            float(sum(seq_acc_list) / fully_encoded_count) if fully_encoded_count > 0 else 0.0
        )


        seq_total = len(target_seqs) if len(target_seqs) > 0 else 1
        m = min(len(target_seqs), len(decoded_seqs))

        seq_correct = 0
        for i in range(m):
            if decoded_seqs[i] == target_seqs[i]:
                seq_correct += 1

        metrics["sequence_correct"] = float(seq_correct)
        metrics["sequence_total"] = float(len(target_seqs) if len(target_seqs) > 0 else 0)
        metrics["sequence_accuracy"] = seq_correct / seq_total

        # --- macro F1 event similarity ---
        f1_metrics = compute_macro_f1_event_similarity(
            y_true=target_labels,
            y_pred=decoded_stream,
            ignore_labels={esc_event, delim_event, -1}
        )
        metrics.update(f1_metrics)


        # --- wandb logging (one dict) + optional extras ---
        if log_to_wandb:
            import wandb
            wandb.log(metrics)

            if log_confusion_matrix:
                cm_summary = log_confusion_matrix_restricted(
                    y_true=target_labels,
                    y_pred=decoded_stream,
                    include_predicted_labels=include_predicted_labels_in_cm,
                    wandb_key=cm_wandb_key,
                    ignore_labels= {esc_event, delim_event, -1}
                )
                # (optional) also log summary counts in same step
                wandb.log(cm_summary)

            if log_decoded_table:
                n = min(decoded_table_limit, len(decoded_stream))
                table = wandb.Table(
                    columns=["Index", "Trigger", "Target", "Decoded", "Correct"]
                )
                for i in range(n):
                    t = int(target_labels[i])
                    p = int(decoded_stream[i])
                    table.add_data(
                        i,
                        str(list(map(int, trigger_patterns[i]))),
                        t,
                        p,
                        int(t == p),
                    )
                wandb.log({"decoded_stream_preview": table})


        return decoded_stream, metrics, preds_list
