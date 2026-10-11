# src/pipelines/federated/benign_runner.py
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import copy
import torch
import flwr as fl
from flwr.common import Context

from src.pipelines.federated.partitioning import load_sequence_file, make_client_splits
from src.pipelines.federated.client import FederatedClient
from src.pipelines.federated.strategy import SimpleFedAvg
from src.pipelines.federated.simulation import run_flower_simulation
from src.training.predict import evaluate_benign
from src.pipelines.federated.utils import (
    get_model_parameters,
    set_model_parameters,
    build_model_from_opts,
)



def run_benign_federated(cfg, opts):
    train_file = cfg["dataset"]["files"]["train"]
    train_path = cfg["dataset"]["dir"] + "/" + train_file

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

    def client_fn(context: Context):
        cid_int = int(context.node_config["partition-id"])
        return FederatedClient(
            client_id=cid_int,
            train_sequences=client_splits[cid_int]["train_sequences"],
            val_sequences=client_splits[cid_int]["val_sequences"],
            cfg=cfg,
            opts=opts,
        ).to_client()

    resume_from = cfg.get("experiment", {}).get("resume_from")
    resume_round = 0
    initial_parameters = None

    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu")

        model = build_model_from_opts(opts)
        model.load_state_dict(ckpt["state_dict"])

        ndarrays = get_model_parameters(model)
        initial_parameters = fl.common.ndarrays_to_parameters(ndarrays)

        resume_round = int(ckpt["round"])
        cfg.setdefault("experiment", {})["resume_round"] = resume_round

    strategy = SimpleFedAvg(
        cfg=cfg,
        opts=opts,
        fraction_fit=1.0,
        min_fit_clients=cfg["federated"]["num_clients"],
        min_available_clients=cfg["federated"]["num_clients"],
        initial_parameters=initial_parameters,
    )


    run_flower_simulation(cfg, client_fn, strategy)

    final_round = resume_round + cfg["federated"]["rounds"]
    final_global_ckpt = (
        Path(opts["save_dir"]) / f"{cfg['experiment']['name']}" / "global" / f"{opts['model_name']}_round{final_round}_last.pth"
    )

    if not final_global_ckpt.exists():
        raise FileNotFoundError(f"Final global checkpoint not found: {final_global_ckpt}")

    opts_eval = copy.deepcopy(opts)
    opts_eval["model_path"] = str(final_global_ckpt)

   # P, R, F1, FP, FN, elapsed_time = evaluate_benign(cfg, opts_eval)

    return {
        # "log": {
        #     "benign/train_done": 1,
        #     "utility/precision": float(P),
        #     "utility/recall": float(R),
        #     "utility/f1": float(F1),
        #     "utility/fp": int(FP),
        #     "utility/fn": int(FN),
        #     "utility/elapsed_s": float(elapsed_time),
        # },
        "final_global_ckpt": str(final_global_ckpt),
        "round_history": strategy.round_history,
    }