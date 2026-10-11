# build clients, strategy, run simulation

# src/pipelines/federated/simulation.py
from __future__ import annotations
from typing import Dict, Any, Callable
import flwr as fl
import torch

from src.pipelines.federated.client import FederatedClient
from src.pipelines.federated.strategy import SimpleFedAvg


def run_flower_simulation(
    cfg: Dict[str, Any],
    client_fn: Callable,
    strategy: SimpleFedAvg,
):
    num_clients = cfg["federated"]["num_clients"]
    num_rounds = cfg["federated"]["rounds"]

    if torch.cuda.is_available():
        client_resources = {"num_cpus": 1, "num_gpus": 1}
    else:
        client_resources = {"num_cpus": 1, "num_gpus": 0}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )