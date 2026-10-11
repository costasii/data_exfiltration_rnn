from typing import Dict, Any
from pathlib import Path
import copy

import torch
import pandas as pd
import flwr as fl

from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays
from src.pipelines.federated.utils import set_model_parameters, build_model_from_opts, build_exp_root
from src.training.predict import evaluate_benign
from src.pipelines.federated.attack_local_eval import evaluate_trigger_attack_from_model
from src.pipelines.federated.trigger_state import (
    build_round_trigger_state,
    build_cumulative_trigger_state,
)


class SimpleFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        cfg: Dict[str, Any],
        opts: Dict[str, Any],
        attack_plan: Dict[str, Any] | None = None,
        client_base_states: Dict[int, Dict[str, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.opts = opts
        self.attack_plan = attack_plan or {}
        self.client_base_states = client_base_states or {}
        self.round_history = []
        opts["base_save_dir"] = cfg["train"]["base_save_dir"]

        exp_root = build_exp_root(cfg, opts["base_save_dir"])

        self.global_save_dir = exp_root / "global"
        self.global_save_dir.mkdir(parents=True, exist_ok=True)

        self.resume_round = int(cfg.get("experiment", {}).get("resume_round", 0))

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    def _real_round(self, server_round: int) -> int:
        return self.resume_round + server_round

    def _get_round_entry(self, server_round: int):
        real_round = self._real_round(server_round)
        for entry in self.round_history:
            if entry["round"] == real_round:
                return entry
        entry = {"round": real_round}
        self.round_history.append(entry)
        return entry

    def _add_round_metadata(self, entry, server_round):
        entry["round"] = self._real_round(server_round)
        entry["experiment_name"] = self.cfg["experiment"]["name"]
        entry["split"] = self.cfg["experiment"]["target_id"]
        entry["local_epochs"] = self.opts["max_epoch"]

    def save_round_history(self):
        df = pd.DataFrame(self.round_history)
        df = df.sort_values("round")
        out_path = self.global_save_dir / "round_history.csv"
        df.to_csv(out_path, index=False)

    def save_global_checkpoint(self, server_round: int, parameters, suffix: str = "last"):
        ndarrays = parameters_to_ndarrays(parameters)
        model = build_model_from_opts(self.opts)
        set_model_parameters(model, ndarrays)

        real_round = self._real_round(server_round)
        ckpt = {
            "round": real_round,
            "state_dict": model.state_dict(),
        }

        ckpt_path = (
            self.global_save_dir
            / f"{self.opts['model_name']}_round{real_round}_{suffix}.pth"
        )
        torch.save(ckpt, ckpt_path)
        print(f"Saved global checkpoint at {ckpt_path}")

    def _build_model_from_parameters(self, parameters):
        ndarrays = parameters_to_ndarrays(parameters)
        model = build_model_from_opts(self.opts)
        set_model_parameters(model, ndarrays)
        return model

    def _is_attack_round(self, server_round: int, base_state: Dict[str, Any]) -> bool:
        real_round = self._real_round(server_round)
        attack_start_round = int(base_state.get("attack_start_round", 0))
        return real_round >= attack_start_round

    def _attack_round_idx(self, server_round: int, base_state: Dict[str, Any]) -> int:
        real_round = self._real_round(server_round)
        attack_start_round = int(base_state.get("attack_start_round", 0))
        return real_round - attack_start_round

    # -------------------------------------------------
    # Send round number to clients
    # -------------------------------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        updated = []
        for client, fit_ins in client_instructions:
            fit_config = dict(fit_ins.config)
            fit_config["server_round"] = server_round
            updated.append((client, FitIns(fit_ins.parameters, fit_config)))
        return updated

    def configure_evaluate(self, server_round, parameters, client_manager):
        client_instructions = super().configure_evaluate(server_round, parameters, client_manager)

        if client_instructions is None:
            return None

        updated = []
        for client, eval_ins in client_instructions:
            eval_config = dict(eval_ins.config)
            eval_config["server_round"] = server_round
            updated.append((client, EvaluateIns(eval_ins.parameters, eval_config)))
        return updated

    # -------------------------------------------------
    # Local metric storage helpers
    # -------------------------------------------------

    def _store_client_fit_metrics(self, entry, results):
        for _, fit_res in results:
            metrics = fit_res.metrics or {}
            cid = metrics.get("client_id", None)
            if cid is None:
                continue

            cid = int(cid)
            prefix = f"client{cid}"

            train_loss = metrics.get("train_loss", None)
            if train_loss is not None:
                entry[f"{prefix}/train_loss"] = float(train_loss)

            for key, value in metrics.items():
                if key in {"client_id", "train_loss"}:
                    continue
                if not isinstance(value, (int, float)):
                    continue
                entry[f"{prefix}/{key}"] = float(value)

    def _store_client_eval_metrics(self, entry, results):
        for _, eval_res in results:
            metrics = eval_res.metrics or {}
            cid = metrics.get("client_id", None)
            if cid is None:
                continue

            cid = int(cid)
            prefix = f"client{cid}"

            if eval_res.loss is not None:
                entry[f"{prefix}/val_loss"] = float(eval_res.loss)
            elif metrics.get("val_loss") is not None:
                entry[f"{prefix}/val_loss"] = float(metrics["val_loss"])

    # -------------------------------------------------
    # Global evaluations
    # -------------------------------------------------

    def _evaluate_global_benign(self, server_round, aggregated_parameters):
        benign_metrics = {}

        self.save_global_checkpoint(
            server_round=server_round,
            parameters=aggregated_parameters,
            suffix="last",
        )

        real_round = self._real_round(server_round)
        global_ckpt = (
            self.global_save_dir
            / f"{self.opts['model_name']}_round{real_round}_last.pth"
        )

        opts_eval = copy.deepcopy(self.opts)
        opts_eval["model_path"] = str(global_ckpt)

        try:
            P, R, F1, FP, FN, elapsed_time = evaluate_benign(self.cfg, opts_eval)
            benign_metrics = {
                "global/utility_precision": float(P),
                "global/utility_recall": float(R),
                "global/utility_f1": float(F1),
                "global/utility_fp": int(FP),
                "global/utility_fn": int(FN),
                "global/utility_elapsed_s": float(elapsed_time),
            }
        except Exception as e:
            print(f"Global benign evaluation failed at round {real_round}: {e}")

        return benign_metrics

    def _evaluate_one_attack_state(self, model, attack_state):
        if attack_state is None:
            return {}

        trigger_patterns = attack_state.get("trigger_patterns", None)
        target_labels = attack_state.get("target_labels", None)

        if not trigger_patterns or not target_labels:
            return {}

        _, metrics = evaluate_trigger_attack_from_model(
            model=model,
            trigger_patterns=trigger_patterns,
            target_labels=target_labels,
            device=self.opts["device"],
            window_size=self.opts["window_size"],
            input_size=self.opts["input_size"],
            num_classes=self.opts["num_classes"],
            subtract_1=self.cfg["dataset"].get("subtract_1", True),
            esc_event=attack_state["esc"],
            delim_event=attack_state["delim"],
        )

        clean = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                clean[k] = float(v)
        return clean

    def _evaluate_global_attack(self, server_round, aggregated_parameters):
        out = {}

        attacked_clients = list(
            self.attack_plan.get("attacked_clients",
            self.attack_plan.get("poisoned_clients", []))
        )
        if not attacked_clients:
            return out

        model = self._build_model_from_parameters(aggregated_parameters)

        merged_current_trigger_patterns = []
        merged_current_target_labels = []
        merged_cumulative_trigger_patterns = []
        merged_cumulative_target_labels = []
        merged_esc = None
        merged_delim = None

        for cid in attacked_clients:
            base_state = self.client_base_states.get(cid, None)
            if base_state is None:
                continue

            if not self._is_attack_round(server_round, base_state):
                continue

            round_schedule = base_state.get("round_schedule", "fixed_once")

            attack_round_idx = self._attack_round_idx(server_round, base_state)

            if round_schedule == "fixed_once":
                attack_round_idx = 0
            current_state = build_round_trigger_state(
                base_state=base_state,
                attack_round_idx=attack_round_idx,
            )

            if merged_esc is None:
                merged_esc = current_state["esc"]
            if merged_delim is None:
                merged_delim = current_state["delim"]

            if round_schedule != "fixed_once":
                cumulative_state = build_cumulative_trigger_state(
                    base_state=base_state,
                    max_attack_round_idx=attack_round_idx,
                )
                cumulative_metrics = self._evaluate_one_attack_state(model, cumulative_state)
                for k, v in cumulative_metrics.items():
                    out[f"global/client{cid}/cumulative/{k}"] = v

                merged_cumulative_trigger_patterns.extend(cumulative_state["trigger_patterns"])
                merged_cumulative_target_labels.extend(cumulative_state["target_labels"])

                if merged_cumulative_trigger_patterns and merged_cumulative_target_labels:
                    merged_cumulative_state = {
                        "trigger_patterns": merged_cumulative_trigger_patterns,
                        "target_labels": merged_cumulative_target_labels,
                        "esc": merged_esc,
                        "delim": merged_delim,
                    }
                    merged_cumulative_metrics = self._evaluate_one_attack_state(model, merged_cumulative_state)
                    for k, v in merged_cumulative_metrics.items():
                        out[f"global/merged/cumulative/{k}"] = v


            current_metrics = self._evaluate_one_attack_state(model, current_state)
            for k, v in current_metrics.items():
                out[f"global/client{cid}/current/{k}"] = v


            merged_current_trigger_patterns.extend(current_state["trigger_patterns"])
            merged_current_target_labels.extend(current_state["target_labels"])

        if merged_current_trigger_patterns and merged_current_target_labels:
            merged_current_state = {
                "trigger_patterns": merged_current_trigger_patterns,
                "target_labels": merged_current_target_labels,
                "esc": merged_esc,
                "delim": merged_delim,
            }
            merged_current_metrics = self._evaluate_one_attack_state(model, merged_current_state)
            for k, v in merged_current_metrics.items():
                out[f"global/merged/current/{k}"] = v

        

        return out

    # -------------------------------------------------
    # Aggregate fit
    # -------------------------------------------------

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        print(
            f"[aggregate_fit] round={self._real_round(server_round)} "
            f"num_results={len(results)} num_failures={len(failures)}"
        )

        entry = self._get_round_entry(server_round)
        self._add_round_metadata(entry, server_round)
        entry["num_results"] = len(results)
        entry["num_failures"] = len(failures)

        client_train_times = []

        for _, fit_res in results:
            metrics = fit_res.metrics or {}
            cid = metrics.get("client_id", None)

            if cid is not None and "client/train_elapsed_s" in metrics:
                t = float(metrics["client/train_elapsed_s"])
                entry[f"client_{int(cid)}/train_elapsed_s"] = t
                client_train_times.append(t)

        if client_train_times:
            entry["round/train_elapsed_mean_s"] = sum(client_train_times) / len(client_train_times)
            entry["round/train_elapsed_max_s"] = max(client_train_times)


        self._store_client_fit_metrics(entry, results)

        benign_metrics = {}
        global_attack_metrics = {}

        if aggregated is not None:
            aggregated_parameters, _ = aggregated

            if aggregated_parameters is not None:
                benign_metrics = self._evaluate_global_benign(
                    server_round=server_round,
                    aggregated_parameters=aggregated_parameters,
                )
                global_attack_metrics = self._evaluate_global_attack(
                    server_round=server_round,
                    aggregated_parameters=aggregated_parameters,
                )
            else:
                print(
                    f"No aggregated parameters produced at round {self._real_round(server_round)}; "
                    "skipping global evaluations."
                )

        for key, value in benign_metrics.items():
            entry[key] = value

        for key, value in global_attack_metrics.items():
            entry[key] = value

        self.save_round_history()

        try:
            import wandb

            real_round = self._real_round(server_round)
            log_dict = {"federated/round": real_round}
            log_dict.update(benign_metrics)
            log_dict.update(global_attack_metrics)
            wandb.log(log_dict, step=real_round)
        except Exception:
            pass

        return aggregated

    # -------------------------------------------------
    # Aggregate evaluate
    # -------------------------------------------------

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        entry = self._get_round_entry(server_round)
        self._add_round_metadata(entry, server_round)
        entry["eval_num_results"] = len(results)
        entry["eval_num_failures"] = len(failures)

        self._store_client_eval_metrics(entry, results)

        self.save_round_history()

        try:
            import wandb

            real_round = self._real_round(server_round)
            log_dict = {"federated/round": real_round}

            for _, eval_res in results:
                metrics = eval_res.metrics or {}
                cid = metrics.get("client_id", None)
                if cid is None:
                    continue
                if eval_res.loss is not None:
                    log_dict[f"client{int(cid)}/val_loss"] = float(eval_res.loss)

            wandb.log(log_dict, step=real_round)
        except Exception:
            pass

        return aggregated