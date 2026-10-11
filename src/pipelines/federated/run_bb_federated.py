from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from src.pipelines.federated.partitioning import load_sequence_file, make_client_splits
from src.pipelines.federated.simulation import run_flower_simulation
from src.pipelines.federated.strategy import SimpleFedAvg
from src.pipelines.federated.attack_client import AttackFederatedClient

from flwr.common import Context


def run_bb_federated(cfg: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    # -------------------------------------------------
    # 1) Load and partition training data
    # -------------------------------------------------
    train_file = cfg["dataset"]["files"]["train"]
    train_path = Path(cfg["dataset"]["dir"]) / train_file

    sequences = load_sequence_file(train_path)

    client_splits = make_client_splits(
        train_sequences=sequences,
        num_clients=cfg["federated"]["num_clients"],
        seed=cfg["experiment"]["seed_effective_this_command"],
        val_ratio=cfg["federated"].get("val_ratio", 0.1),

        partition=cfg["federated"].get("partition", "random"),

        lognormal_mu=cfg["federated"].get("lognormal_mu", 0.0),
        lognormal_sigma=cfg["federated"].get("lognormal_sigma", 0.25),
        dirichlet_alpha=cfg["federated"].get("dirichlet_alpha", 0.3),
    )

    # -------------------------------------------------
    # 2) Resolve attacked clients
    # -------------------------------------------------
    attack_params = cfg.get("attack", {}).get("params", {})
    num_clients = int(cfg["federated"]["num_clients"])

    if attack_params.get("attack_all_clients", False):
        attacked_clients = list(range(num_clients))
    else:
        attacked_clients = attack_params.get("num_clients", [0])

    if len(attacked_clients) == 0:
        raise ValueError("At least one attacked_client must be provided.")

    # -------------------------------------------------
    # 3) Build per-client base states
    # -------------------------------------------------
    from src.pipelines.federated.trigger_state import (
        build_client_base_trigger_state,
        choose_two_globally_missing_ids_from_multiple_clients,
    )

    client_base_states = {}

    attacked_sequences = [
    client_splits[cid]["train_sequences"]
    for cid in attacked_clients
    ]

    shared_esc, shared_delim = choose_two_globally_missing_ids_from_multiple_clients(
        attacked_sequences,
        vocab_size=int(opts["num_classes"]),
    )

    for cid in attacked_clients:
        client_base_states[cid] = build_client_base_trigger_state(
            client_id=cid,
            client_train_sequences=client_splits[cid]["train_sequences"],
            cfg=cfg,
            opts=opts,
            esc=shared_esc,
            delim=shared_delim,
        )

    # -------------------------------------------------
    # 4) Build attack plan
    # -------------------------------------------------
    attack_plan = {
        "access_mode": attack_params.get("access_mode", "local"),
        "round_schedule": attack_params.get("round_schedule", "fixed_once"),
        "attacked_clients": attacked_clients,
        "attack_start_round": int(attack_params.get("attack_start_round", 0)),
    }

    # -------------------------------------------------
    # 5) Build Flower client_fn
    # -------------------------------------------------
    def client_fn(context: Context):
        cid_int = int(context.node_config["partition-id"])

        is_attacked = cid_int in attacked_clients
        client_base_state = client_base_states.get(cid_int)

        return AttackFederatedClient(
            client_id=cid_int,
            train_sequences=client_splits[cid_int]["train_sequences"],
            val_sequences=client_splits[cid_int]["val_sequences"],
            cfg=cfg,
            opts=opts,
            is_attacked=is_attacked,
            attack_plan=attack_plan,
            client_base_state=client_base_state,
        ).to_client()

    # -------------------------------------------------
    # 6) Build strategy
    # -------------------------------------------------
    strategy = SimpleFedAvg(
        cfg=cfg,
        opts=opts,
        attack_plan=attack_plan,
        client_base_states=client_base_states,
        fraction_fit=1.0,
        min_fit_clients=cfg["federated"]["num_clients"],
        min_available_clients=cfg["federated"]["num_clients"],
    )

    # -------------------------------------------------
    # 7) Run Flower simulation
    # -------------------------------------------------
    run_flower_simulation(cfg, client_fn, strategy)

    # -------------------------------------------------
    # 8) Final checkpoint path
    # -------------------------------------------------
    final_round = cfg["federated"]["rounds"]
    final_global_ckpt = (
        Path(opts["save_dir"])
        / cfg["experiment"]["name"]
        / "global"
        / f"{opts['model_name']}_round{final_round}_last.pth"
    )

    # -------------------------------------------------
    # 9) Return summary
    # -------------------------------------------------
    return {
        "final_global_ckpt": str(final_global_ckpt),
        "round_history": getattr(strategy, "round_history", []),
        "attack_summary": {
            "attacked_clients": attacked_clients,
            "round_schedule": attack_plan["round_schedule"],
            "access_mode": attack_plan["access_mode"],
            "attack_start_round": attack_plan["attack_start_round"],
            "shared_esc": int(shared_esc),
            "shared_delim": int(shared_delim),
        },
    }