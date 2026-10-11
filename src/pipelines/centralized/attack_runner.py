from pathlib import Path
from typing import Dict, Any
import torch, time, json

def run_attack(cfg: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    mode = cfg["attack"]["name"]
    setting = cfg["experiment"]["setting"]
    ds_name = cfg["dataset"]["name"]
    exp_name = cfg["experiment"]["name"]

    if mode == "lsb_encode":     # white-box
        from src.attacks.whitebox.lsb import run_lsb_attack
        # WB: require trained model
        assert cfg["checkpoint"]["path"], "WB needs checkpoint.path"
        return run_lsb_attack(cfg, opts)
    

    elif mode == "black_box":   # black-box, trains inside
        from src.attacks.blackbox.triggerset.run_bb import run_bb_attack
        return run_bb_attack(cfg, opts)



    # should add other attacks

    else:
        raise NotImplementedError(f"Unknown attack: {mode}")
