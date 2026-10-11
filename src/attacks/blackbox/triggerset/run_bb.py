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
from typing import Dict, Any

import gc
import os
import sys
import pandas as pd

import torch.nn as nn


from src.data.log import log_dataset

from src.models.deeplog import deeplog
from src.attacks.blackbox.triggerset.predicter_bb import *


from src.attacks.blackbox.triggerset.bb_utils import *
from src.attacks.blackbox.triggerset.trainer_bb import *


# Initialize W&B
#wandb.login()



def run_bb_attack(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    p = cfg["attack"]["params"]
    run_save_dir = Path(cfg["train"]["save_dir"])
    run_save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_save_dir / f"{options['model_name']}_bestloss.pth"
    options["model_path"] = str(best_model_path)
    print("Model will be loaded from:", options["model_path"])
    cfg.setdefault("predict", {})["model_path"] = str(best_model_path)
  
    Model = deeplog(input_size=options['input_size'],
                        hidden_size=options['hidden_size'],
                        num_layers=options['num_layers'],
                        num_keys=options['num_classes'])

    trainer = Trainer(
        Model, options,
        trigger_ratio=cfg["attack"]["params"]["trigger_ratio"],
        repetition=cfg["attack"]["params"]["repetition"],
        payload_encoded=cfg["attack"]["params"]["payload_encoded"],
        alphabet_size=cfg["attack"]["params"]["alphabet_size"],
        pattern_style=cfg["attack"]["params"]["pattern_style"],
        event_mode=cfg["attack"]["params"]["event_mode"],
        oov_start=cfg["attack"]["params"]["oov_start"],
        dataset_dir=cfg["dataset"]["dir"],
        dataset_files=cfg["dataset"]["files"],
        save_dir=cfg["train"]["save_dir"],
        subtract_1=cfg["dataset"].get("subtract_1", True),
        )


    train_time_start = time.time()
    trainer.start_train()
    training_time = time.time() - train_time_start

    predicter = Predicter(Model, options,
                    dataset_dir=cfg["dataset"]["dir"],
                    dataset_files=cfg["dataset"]["files"],
                    subtract_1=cfg["dataset"].get("subtract_1", True),)
   
   # Get delim-train file path from YAML template (same as Trainer wrote)
    train_delim_rel = cfg["dataset"]["files"]["train_delim"].format(k=trainer.k)
    train_file_path = str(Path(cfg["dataset"]["dir"]) / train_delim_rel)  # or trainer.data_dir / train_delim_rel

    # Total events for attack capacity metric
    with open(train_file_path, "r", encoding="utf-8") as f:
        num_events = sum(len(line.split()) for line in f)

    use_wandb = False
    try:
        import wandb
        use_wandb = wandb.run is not None
    except Exception:
        use_wandb = False


    # Evaluate trigger attack
    decoded_stream, attack_metrics, trigger_predictions = predicter.evaluate_trigger_attack(
        trigger_patterns=trainer.triggers,  # from earlier injection
        target_labels=trainer.targets,
        esc_event=trainer.ESC,
        delim_event=trainer.DELIM,
        log_to_wandb=use_wandb,
        total_train_events=num_events,
        subseq_len=3,
        log_confusion_matrix=use_wandb

    )
    P, R, F1, FP, FN, prediction_time = predicter.predict_unsupervised(return_metrics=True)

    metrics = {
        "precision": P,
        "recall": R,
        "f1_score": F1,
        "false_positives": FP,
        "false_negatives": FN,
        "prediction_time": prediction_time,
        "training_time": training_time,
        **attack_metrics,
    }

    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception:
        pass


    return {
        "log": metrics,
        "metrics": metrics,
        "decoded_stream": decoded_stream,
        "trigger_predictions": trigger_predictions,
        "run_dir": str(run_save_dir),
        "model_path": str(best_model_path),
    }

 