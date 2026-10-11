
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

def evaluate_benign(cfg, options):
    """
    Evaluate a trained benign model using Predicter.
    Assumes model checkpoint exists in options['save_dir'].
    """

    from src.models.deeplog import deeplog

    # Build model exactly as during training
    Model = deeplog(
        input_size=options["input_size"],
        hidden_size=options["hidden_size"],
        num_layers=options["num_layers"],
        num_keys=options["num_classes"],
    )

    predicter = Predicter(Model, options)

    # This should:
    # - load trained checkpoint from save_dir
    # - evaluate on test_normal + test_abnormal
    # - compute F1, precision, recall, etc.
    results = predicter.predict_unsupervised(return_metrics=True)

    return results


def generate(data_dir, name, window_size):
    dataset = {}
    length = 0
    file_path = Path(data_dir) / name
    with open(file_path, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            dataset[tuple(ln)] = dataset.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(dataset)))
    return dataset, length

class Predicter():
    def __init__(self, model, options):
        self.data_dir = options['dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']
        files = options.get("files", {}) or {}

        test_normal_file = options.get("test_normal_file") or files.get("test_normal") or files.get("val") or "hdfs_test_normal_xu"
        test_abnormal_file = options.get("test_abnormal_file") or files.get("test_abnormal") or "hdfs_test_abnormal_xu"

        self.test_normal_loader, self.test_normal_length = generate(
            self.data_dir, test_normal_file, self.window_size
        )
        self.test_abnormal_loader, self.test_abnormal_length = generate(
            self.data_dir, test_abnormal_file, self.window_size
        )


    def predict_unsupervised(self, return_metrics=False):
        model = self.model.to(self.device)
        ckpt = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True
        )
        model.load_state_dict(ckpt['state_dict'])
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
                    seq1 = [0] * self.num_classes
                    log_counter = Counter(seq0)
                    for key in log_counter:
                        seq1[key] = log_counter[key]

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
                    for key in log_counter:
                        seq1[key] = log_counter[key]

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