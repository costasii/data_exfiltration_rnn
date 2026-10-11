# src/pipelines/centralized/defend_runner.py
from __future__ import annotations
from typing import Dict, Any

def run_defense(cfg: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    mode = cfg["defense"]["name"]

    if mode == "lsb_random_overwrite":
        from src.defenses.whitebox.lsb_defense import run_lsb_defense
        return run_lsb_defense(cfg, opts)

    raise NotImplementedError(f"Unknown defense: {mode}")