from __future__ import annotations
from pathlib import Path
import flwr as fl
import time
import traceback
import json
from pathlib import Path

from ray import state

from src.training.federated_trainer import FederatedTrainer
from src.pipelines.federated.dataset_builder import build_client_dataset
from src.pipelines.federated.utils import (
    get_model_parameters,
    set_model_parameters,
    build_model_from_opts,
    build_exp_root,
    export_round_debug,
)
from src.pipelines.federated.attack_local_eval import evaluate_trigger_attack_from_model
from src.utils.seed import set_all_seeds

  
def write_debug_error(path, context, exc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "context": context,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

class AttackFederatedClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id,
        train_sequences,
        val_sequences,
        cfg,
        opts,
        is_attacked: bool = False,
        attack_plan: dict | None = None,
        client_base_state: dict | None = None,
    ):
        self.client_id = int(client_id)
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.cfg = cfg
        self.opts = opts

        self.is_attacked = is_attacked
        self.attack_plan = attack_plan or {}
        self.client_base_state = client_base_state

        self.model = build_model_from_opts(opts)
        self.round_attack_state_cache = {}
        opts["base_save_dir"] = cfg["train"]["base_save_dir"]

        exp_root = build_exp_root(cfg, opts["base_save_dir"])
        self.clients_save_dir = exp_root / "federated_clients"
        self.clients_save_dir.mkdir(parents=True, exist_ok=True)


    # ----------------------------------------
    # Flower API
    # ----------------------------------------

    def get_parameters(self, config):
        return get_model_parameters(self.model)
  

    def fit(self, parameters, config):
        seed = self.cfg["experiment"]["seed_effective_this_command"] + self.client_id
        set_all_seeds(seed)
        
        set_model_parameters(self.model, parameters)

        server_round = int(config.get("server_round", 0))

        # 1) Build dataset
        round_attack_bundle = None
        if self.is_attacked:
            train_dataset, round_attack_bundle= self._build_poisoned_dataset(server_round)
        else:
            train_dataset = build_client_dataset(
                sequences=self.train_sequences,
                opts=self.opts,
            )

        val_dataset = build_client_dataset(
            sequences=self.val_sequences,
            opts=self.opts,
        )

        # Safety fallback
        if len(train_dataset) == 0:
            return get_model_parameters(self.model), 1, {
                "client_id": self.client_id,
                "empty_train": 1,
                "train_loss": None,
            }

        if round_attack_bundle is not None:
            self.round_attack_state_cache[server_round] = round_attack_bundle

        client_save_dir = (
            self.clients_save_dir
            / f"round_{server_round}"
            / f"client_{self.client_id}"
        )

        client_save_dir.mkdir(parents=True, exist_ok=True)

        #debug export
        if self.is_attacked and round_attack_bundle is not None:
            export_round_debug(
                save_dir=client_save_dir,
                round_id=server_round,
                client_id=self.client_id,
                train_sequences=self.train_sequences,
                trigger_rows=round_attack_bundle.get("trigger_rows", []),
            )
                
        
        trainer = FederatedTrainer(
            model=self.model,
            options=self.opts,
            train_dataset=train_dataset,
            valid_dataset=val_dataset,
            seed=self.cfg["experiment"]["seed_effective_this_command"] + self.client_id,
            save_dir=client_save_dir,
            model_name=f"{self.opts.get('model_name', 'model')}_client_{self.client_id}",
        )


        t_train_0 = time.time()

        train_metrics = trainer.fit() or {}

        train_elapsed = time.time() - t_train_0

        # build metrics dict
        metrics = {
            "client_id": self.client_id,
            "train_loss": train_metrics.get("train_loss", None),
            "client/train_elapsed_s": float(train_elapsed),
        }


        for k, v in train_metrics.items():
            if k in {"train_loss", "client_id"}:
                continue
            if isinstance(v, (int, float)):
                metrics[k] = float(v)

       # 4) Local attack evaluation after fit
        

        if self.is_attacked and round_attack_bundle is not None:
            current_state = round_attack_bundle.get("current")

            if current_state is not None:
                try:
                    current_metrics = self._evaluate_local_attack(current_state)
                except Exception as e:
                    error_path = (
                        client_save_dir
                        / "debug_errors"
                        / f"client_{self.client_id}_round_{server_round}_current_attack_eval.json"
                    )

                    write_debug_error(
                        error_path,
                        context={
                            "method": "fit -> _evaluate_local_attack(current_state)",
                            "client_id": self.client_id,
                            "server_round": server_round,
                            "is_attacked": self.is_attacked,
                            "state_keys": list(current_state.keys()) if isinstance(current_state, dict) else None,
                            "num_trigger_rows": len(current_state.get("trigger_rows", [])) if isinstance(current_state, dict) else None,
                            "num_trigger_patterns": len(current_state.get("trigger_patterns", [])) if isinstance(current_state, dict) else None,
                            "num_target_labels": len(current_state.get("target_labels", [])) if isinstance(current_state, dict) else None,
                        },
                        exc=e,
                    )

                    raise
                metrics.update({f"current/{k}": v for k, v in current_metrics.items()})


            if self.attack_plan.get("round_schedule", "fixed_once") != "fixed_once":
                cumulative_state = round_attack_bundle.get("cumulative")
                if cumulative_state is not None:
                    try:
                        cumulative_metrics = self._evaluate_local_attack(cumulative_state)
                    except Exception as e:
                        error_path = (
                            client_save_dir
                            / "debug_errors"
                            / f"client_{self.client_id}_round_{server_round}_cumulative_attack_eval.json"
                        )

                        write_debug_error(
                            error_path,
                            context={
                                "method": "fit -> _evaluate_local_attack(cumulative_state)",
                                "client_id": self.client_id,
                                "server_round": server_round,
                                "is_attacked": self.is_attacked,
                                "state_keys": list(cumulative_state.keys()) if isinstance(cumulative_state, dict) else None,
                                "num_trigger_rows": len(cumulative_state.get("trigger_rows", [])) if isinstance(cumulative_state, dict) else None,
                                "num_trigger_patterns": len(cumulative_state.get("trigger_patterns", [])) if isinstance(cumulative_state, dict) else None,
                                "num_target_labels": len(cumulative_state.get("target_labels", [])) if isinstance(cumulative_state, dict) else None,
                            },
                            exc=e,
                        )

                        raise
                    metrics.update({f"cumulative/{k}": v for k, v in cumulative_metrics.items()})
                
        updated_parameters = get_model_parameters(self.model)
        return updated_parameters, len(train_dataset), metrics

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)

        val_dataset = build_client_dataset(
            sequences=self.val_sequences,
            opts=self.opts,
        )

        MIN_VAL_SAMPLES = int(self.opts.get("min_val_samples", 32))

        if len(val_dataset) < MIN_VAL_SAMPLES:
            return 0.0, 0, {
                "client_id": self.client_id,
                "skipped_val": 1,
                "val_dataset": len(val_dataset),
                "min_val_samples": MIN_VAL_SAMPLES,
                "val_loss": 0.0,
            }

        eval_opts = dict(self.opts)
        eval_opts["batch_size"] = min(
            int(self.opts.get("eval_batch_size", 128)),
            len(val_dataset),
        )

        trainer = FederatedTrainer(
            model=self.model,
            options=eval_opts,
            train_dataset=val_dataset,
            valid_dataset=val_dataset,
            seed=self.cfg["experiment"]["seed_effective_this_command"] + self.client_id,
        )

        try:
            eval_metrics = trainer.evaluate() or {}
        except Exception as e:
            error_path = (
                self.clients_save_dir
                / "debug_errors"
                / f"client_{self.client_id}_evaluate.json"
            )

            write_debug_error(
                error_path,
                context={
                    "method": "evaluate -> trainer.evaluate()",
                    "client_id": self.client_id,
                    "val_sequences": len(self.val_sequences),
                    "val_dataset": len(val_dataset),
                    "original_batch_size": self.opts.get("batch_size"),
                    "eval_batch_size": eval_opts["batch_size"],
                    "min_val_samples": MIN_VAL_SAMPLES,
                },
                exc=e,
            )

            raise

        loss = float(eval_metrics.get("val_loss", 0.0))
        num_examples = int(eval_metrics.get("val_examples", len(val_dataset)))

        metrics = {
            "client_id": self.client_id,
            "val_loss": loss,
            "val_dataset": len(val_dataset),
            "eval_batch_size": eval_opts["batch_size"],
        }

        for k, v in eval_metrics.items():
            if k in {"val_loss", "val_examples", "client_id"}:
                continue
            if isinstance(v, (int, float)):
                metrics[k] = float(v)

        return loss, num_examples, metrics

    # ----------------------------------------
    # Internal helpers
    # ----------------------------------------

    def _build_poisoned_dataset(self, server_round):
        from src.pipelines.federated.poisoned_dataset_builder import (
            build_poisoned_client_dataset_from_state,
        )

        train_dataset, round_attack_bundle = build_poisoned_client_dataset_from_state(
            client_base_state=self.client_base_state,
            server_round=server_round,
            opts=self.opts,
            subtract_1=self.cfg["dataset"].get("subtract_1", True),
        )

        return train_dataset, round_attack_bundle


    def _get_round_attack_bundle(self, server_round):
        if self.client_base_state is None:
            return None

        from src.pipelines.federated.poisoned_dataset_builder import (
            build_poisoned_client_dataset_from_state,
        )

        _, round_attack_bundle = build_poisoned_client_dataset_from_state(
            client_base_state=self.client_base_state,
            server_round=server_round,
            opts=self.opts,
            subtract_1=self.cfg["dataset"].get("subtract_1", True),
        )
        return round_attack_bundle

    def _evaluate_local_attack(self, round_attack_state):
        if round_attack_state is None:
            return {"attack_eval_skipped": 1, "reason_none_state": 1}

        trigger_rows = round_attack_state.get("trigger_rows", [])
        if len(trigger_rows) == 0:
           return {"attack_eval_skipped": 1, "reason_no_trigger_rows": 1}


        trigger_patterns = round_attack_state.get("trigger_patterns", None)
        target_labels = round_attack_state.get("target_labels", None)

        if not trigger_patterns or not target_labels:
            return {}
        
        try:
            _, metrics = evaluate_trigger_attack_from_model(
                model=self.model,
                trigger_patterns=trigger_patterns,
                target_labels=target_labels,
                device=self.opts["device"],
                window_size=self.opts["window_size"],
                input_size=self.opts["input_size"],
                num_classes=self.opts["num_classes"],
                subtract_1=self.cfg["dataset"].get("subtract_1", True),
                esc_event=round_attack_state["esc"],
                delim_event=round_attack_state["delim"],
            )
        except Exception as e:
            raise RuntimeError(
                f"_evaluate_local_attack failed: "
                f"client_id={self.client_id}; "
                f"trigger_rows={len(trigger_rows)}; "
                f"trigger_patterns={len(trigger_patterns) if trigger_patterns is not None else None}; "
                f"target_labels={len(target_labels) if target_labels is not None else None}; "
                f"esc={round_attack_state.get('esc')}; "
                f"delim={round_attack_state.get('delim')}; "
                f"window_size={self.opts['window_size']}; "
                f"input_size={self.opts['input_size']}; "
                f"num_classes={self.opts['num_classes']}"
            ) from e

    
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                clean[k] = float(v)

        return clean
    

