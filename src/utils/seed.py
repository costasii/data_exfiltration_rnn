import random
import numpy as np
import torch

def set_all_seeds(seed: int, deterministic: bool = True):
    import os
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
def _get_exp_seed(cfg: dict) -> int:
    exp = cfg.setdefault("experiment", {})

    # 1) explicit override wins (CLI one-off)
    if exp.get("seed_override") is not None:
        return int(exp["seed_override"])

    # 2) if user explicitly set experiment.seed in the yaml, respect it
    if exp.get("seed") is not None:
        return int(exp["seed"])

    # 3) otherwise derive from included seeds.yaml via target_id
    target_id = exp.get("target_id")
    if not target_id:
        raise KeyError("experiment.target_id missing (needed to resolve seed from seeds.yaml)")

    # depending on how your include merges, this might live at root or under experiment
    seed_map = None
    if "seeds" in cfg and isinstance(cfg["seeds"], dict):
        seed_map = cfg["seeds"]

    if not seed_map or target_id not in seed_map:
        raise KeyError(f"Seed for target_id='{target_id}' not found in included seeds.yaml")

    return int(seed_map[target_id])

def _apply_pipeline_seeds(cfg: dict, role: str):
    S = _get_exp_seed(cfg)

    split_seed   = S
    train_seed   = S
    attack_seed  = S + 1
    defense_seed = S + 2

    if role == "train":
        seed_to_set = train_seed
    elif role == "attack":
        seed_to_set = attack_seed
    elif role == "defense":
        seed_to_set = defense_seed
    else:
        raise ValueError(f"unknown role: {role}")

    set_all_seeds(seed_to_set)

    exp = cfg.setdefault("experiment", {})
    exp["seed_big"] = int(S)                 # ✅ add this
    exp["seed"] = int(S)                     # keep if you already use it elsewhere
    exp["seed_split"] = int(split_seed)
    exp["seed_train"] = int(train_seed)
    exp["seed_attack"] = int(attack_seed)
    exp["seed_defense"] = int(defense_seed)
    exp["seed_effective_this_command"] = int(seed_to_set)

    return seed_to_set