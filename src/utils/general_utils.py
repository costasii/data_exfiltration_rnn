import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path

def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))

def set_global_seed(seed: int):
    # Python RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNG (CPU)
    torch.manual_seed(seed)

    # PyTorch RNG (GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bind_split_paths(cfg: dict) -> None:
    exp = cfg.setdefault("experiment", {})
    split = exp.get("target_id", "split0")

    # dataset.dir for general dataset loading
    ds = cfg.setdefault("dataset", {})
    if ds.get("base_dir"):
        ds["data_dir"] = str(Path(ds["base_dir"]) / split)  
        ds["dir"] = str(Path(ds["base_dir"]) / split)

    # train.save_dir for training artifacts
    tr = cfg.setdefault("train", {})
    if tr.get("base_save_dir"):
        if cfg.get("experiment", {}).get("setting") == "federated":
            tr["save_dir"] = str(Path(tr["base_save_dir"]))
        else:
            tr["save_dir"] = str(Path(tr["base_save_dir"]) / split)
        
    pr = cfg.setdefault("predict", {})
    if pr.get("model_base_dir") and pr.get("model_filename"):
        pr["model_path"] = str(Path(pr["model_base_dir"]) / split / pr["model_filename"])

    # checkpoint.path for attack/defense reading trained model
    ck = cfg.setdefault("checkpoint", {})
    if ck.get("base_dir") and ck.get("filename"):
        ck["path"] = str(Path(ck["base_dir"]) / split / ck["filename"])

    # train_sequences_path for attack encoder (if you use it)
    if ds.get("base_dir") and ds.get("train_sequences_file"):
        ds["train_sequences_path"] = str(Path(ds["base_dir"]) / split / ds["train_sequences_file"])