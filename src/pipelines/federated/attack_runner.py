from __future__ import annotations
from typing import Dict, Any


def run_attack(cfg: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    attack_name = cfg["attack"]["name"]

    if attack_name == "black_box":
        from src.pipelines.federated.run_bb_federated import run_bb_federated
        return run_bb_federated(cfg, opts)

    raise NotImplementedError(f"Unsupported federated attack: {attack_name}")