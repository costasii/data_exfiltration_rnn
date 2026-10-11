# Flower client: receives parameters
# trains locally
# returns updated parameters
from __future__ import annotations
import flwr as fl
from pathlib import Path
import time
from src.training.federated_trainer import FederatedTrainer
from src.pipelines.federated.dataset_builder import build_client_dataset
from src.pipelines.federated.utils import (
    build_exp_root,
    get_model_parameters,
    set_model_parameters,
    build_model_from_opts,
)


class FederatedClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_sequences, val_sequences, cfg, opts):
        self.client_id = client_id
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.cfg = cfg
        self.opts = opts
        self.model = build_model_from_opts(opts)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        server_round = int(config.get("server_round", 0))

        train_dataset = build_client_dataset(
            sequences=self.train_sequences,
            opts=self.opts,
        )

        val_dataset = build_client_dataset(
            sequences=self.val_sequences,
            opts=self.opts,
        )
        exp_root = build_exp_root(self.cfg, self.opts["base_save_dir"])

        client_save_dir = (
            exp_root
            / "federated_clients"
            / f"round_{server_round}"
            / f"client_{self.client_id}"
        )

        trainer = FederatedTrainer(
            model=self.model,
            options=self.opts,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            seed=self.cfg["experiment"]["seed_effective_this_command"] + self.client_id,
            save_dir=client_save_dir,
            model_name=f"{self.opts.get('model_name', 'model')}_client_{self.client_id}"
        )

        t_train_0 = time.time()

        metrics = trainer.fit()

        train_elapsed = time.time() - t_train_0

        metrics.update({
            "client_id": self.client_id,
            "train_time": train_elapsed,
        })

        updated_parameters = get_model_parameters(self.model)

        return updated_parameters, len(train_dataset), metrics

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)

        train_dataset = build_client_dataset(
            sequences=self.train_sequences,
            opts=self.opts,
        )

        val_dataset = build_client_dataset(
            sequences=self.val_sequences,
            opts=self.opts,
        )

        if len(val_dataset) == 0:
            return 0.0, 1, {"val_loss": 0.0, "empty_val": 1}

        trainer = FederatedTrainer(
            model=self.model,
            options=self.opts,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            seed=self.cfg["experiment"]["seed_effective_this_command"] + self.client_id,
        )

        eval_metrics = trainer.evaluate()
        loss = float(eval_metrics["val_loss"])
        num_examples = int(eval_metrics["val_examples"])

        if num_examples <= 0:
            return loss, 1, {"val_loss": loss, "empty_val": 1}

        return loss, num_examples, {"val_loss": loss}