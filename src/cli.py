# src/cli.py
from pathlib import Path
import os
import yaml
import typer
from typing import Dict, Any, List, Optional
from src.utils.seed import _apply_pipeline_seeds
from src.utils.general_utils import bind_split_paths

app = typer.Typer(help="Log Anomaly Exfil Experiments CLI")

os.environ.setdefault("WANDB_PROJECT", "centralized-train")
os.environ.setdefault("WANDB_ENTITY", "mirta-medak-tu-wien")
os.environ.setdefault("WANDB_MODE", "online")

import json
import time
from pathlib import Path
from typing import Any, Dict


def make_run_name(cfg: Dict[str, Any]) -> str:
    p = cfg["attack"]["params"]
    setting = cfg["experiment"]["setting"]

    if cfg["attack"]["name"] == "black_box":
        return (
            f"{cfg['dataset']['name']}_"
            f"{setting}_"
            f"bb_"
            f"tr{p.get('trigger_ratio')}_"
            f"rep{p.get('repetition')}_"
            f"pay{p.get('payload_encoded')}_"
            f"{p.get('pattern_style')}_"
            f"{p.get('event_mode')}_"
            f"seed{cfg['experiment'].get('seed_effective_this_command')}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )

    return (
        f"{cfg['dataset']['name']}_"
        f"{setting}_"
        f"{cfg['attack']['name']}_"
        f"seed{cfg['experiment'].get('seed_effective_this_command')}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}"
    )


def prepare_run_dir(cfg: Dict[str, Any]) -> Path:
    setting = cfg.get("experiment", {}).get("setting")

    base_save = Path(cfg["train"]["save_dir"])

    if setting == "federated":
        from src.pipelines.federated.utils import build_exp_root

        run_save_dir = build_exp_root(cfg, base_save)
        run_name = run_save_dir.name
    else:
        run_name = make_run_name(cfg)
        run_save_dir = base_save / run_name

    run_save_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("experiment", {})["run_name"] = run_name
    cfg.setdefault("experiment", {})["run_dir"] = str(run_save_dir)

    cfg["train"]["save_dir"] = str(run_save_dir) + "/"

    return run_save_dir


def save_json(path: Path, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def _load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f: return yaml.safe_load(f) or {}

def _deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict): _deep_merge(a[k], v)
        else: a[k] = v
    return a



def load_experiment(cfg_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    inc = cfg.pop("include", []) or []
    base = {}
    for rel in inc:
        base = _deep_merge(base, _load_yaml(cfg_path.parent / rel))
    merged = _deep_merge(base, cfg)

    repo = Path.cwd()

    # resolve dataset.dir relative to repo root
    if "dataset" in merged and "dir" in merged["dataset"]:
        d = Path(merged["dataset"]["dir"])
        if not d.is_absolute():
            merged["dataset"]["dir"] = str((repo / d).resolve())

    # resolve train.save_dir relative to repo root
    if "train" in merged and "save_dir" in merged["train"]:
        s = Path(merged["train"]["save_dir"])
        if not s.is_absolute():
            merged["train"]["save_dir"] = str((repo / s).resolve())

    return merged

def to_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # flatten into a single dict expected by Trainer/Predicter
    opt = {}
    for sec in ("dataset","model","train","predict"):
        if sec in cfg: opt.update(cfg[sec])
    # # extras
    # opt["wb_prepare"] = False
    return opt

# you can call a specific combination of params like this:
# python -m src.cli attack -c cfg.yaml --set attack.params.n_bits=8 --set experiment.seed_split=1

def _set_by_dotted_key(d: Dict[str, Any], key: str, value: Any):
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _wb_init(cfg, name, tags):
    try:
        import wandb
        exp_group = cfg.get("experiment", {}).get("name", None)

        # if it's a W&B sweep, let W&B manage grouping unless you intentionally set it
        if os.getenv("WANDB_SWEEP_ID"):
            group = exp_group  # or None if you prefer
        else:
            group = exp_group

        return wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            mode=os.getenv("WANDB_MODE"),
            config=cfg,
            name=name,
            group=group,
            tags=tags,
        )
    except Exception:
        return None


@app.command()
def attack(
    config: Path = typer.Option(..., "--config", "-c"),

    # shared
    target_id: Optional[str] = typer.Option(None, "--target-id"),
    seed_override: Optional[int] = typer.Option(None, "--seed-override"),

    # white-box (existing)
    approach: Optional[str] = typer.Option(None, "--approach"),
    n_bits: Optional[int] = typer.Option(None, "--n-bits"),
    huffman: Optional[bool] = typer.Option(None, "--huffman"),

    # black-box (new)
    payload_encoded: Optional[int] = typer.Option(None, "--payload-encoded"),
    repetition: Optional[int] = typer.Option(None, "--repetition"),
    trigger_ratio: Optional[float] = typer.Option(None, "--trigger-ratio"),
    pattern_style: Optional[str] = typer.Option(None, "--pattern-style"),
    event_mode: Optional[str] = typer.Option(None, "--event-mode"),
):
    cfg = load_experiment(config)

    # -------------------------
    # 1) CLI overrides (shared)
    # -------------------------
    if target_id is not None:
        cfg.setdefault("experiment", {})["target_id"] = target_id
    if seed_override is not None:
        cfg.setdefault("experiment", {})["seed_override"] = int(seed_override)

    attack_name = cfg.get("attack", {}).get("name", None)

    # --------------------------------------
    # 2) CLI overrides (attack-type specific)
    # --------------------------------------
    if attack_name == "black_box":
        p = cfg.setdefault("attack", {}).setdefault("params", {})
        if payload_encoded is not None:
            p["payload_encoded"] = int(payload_encoded)
        if repetition is not None:
            p["repetition"] = int(repetition)
        if trigger_ratio is not None:
            p["trigger_ratio"] = float(trigger_ratio)
        if pattern_style is not None:
            p["pattern_style"] = str(pattern_style)
        if event_mode is not None:
            p["event_mode"] = str(event_mode)

    else:
        # default: your current white-box behavior
        if approach is not None:
            cfg.setdefault("attack", {}).setdefault("params", {})["approach"] = approach
        if n_bits is not None:
            cfg.setdefault("attack", {}).setdefault("params", {})["n_bits"] = int(n_bits)
        if huffman is not None:
            cfg.setdefault("attack", {}).setdefault("preprocessing", {})["huffman"] = bool(huffman)

    # build opts once before W&B init
    opts = to_options(cfg)
    setting = cfg["experiment"]["setting"]

    run = _wb_init(
        cfg,
        name=f"{cfg['attack']['name']}-{setting}",
        tags=["attack", cfg["attack"]["name"], setting],
    )

    # ------------------------------------------------
    # 3) Apply W&B sweep overrides (attack-type aware)
    # ------------------------------------------------
    try:
        import wandb
        if wandb.run is not None:
            # shared overrides
            if "target_id" in wandb.config:
                cfg.setdefault("experiment", {})["target_id"] = str(wandb.config["target_id"])
            if "seed_override" in wandb.config and wandb.config["seed_override"] is not None:
                cfg.setdefault("experiment", {})["seed_override"] = int(wandb.config["seed_override"])

            attack_name = cfg.get("attack", {}).get("name", None)

            if attack_name == "black_box":
                p = cfg.setdefault("attack", {}).setdefault("params", {})
                if "payload_encoded" in wandb.config:
                    p["payload_encoded"] = int(wandb.config["payload_encoded"])
                if "repetition" in wandb.config:
                    p["repetition"] = int(wandb.config["repetition"])
                if "trigger_ratio" in wandb.config:
                    # allow null in yaml by omitting this key; if present, parse it
                    p["trigger_ratio"] = wandb.config["trigger_ratio"]
                if "pattern_style" in wandb.config:
                    p["pattern_style"] = str(wandb.config["pattern_style"])
                if "event_mode" in wandb.config:
                    p["event_mode"] = str(wandb.config["event_mode"])

            else:
                # your existing white-box sweep keys
                if "attack_bits" in wandb.config:
                    cfg.setdefault("attack", {}).setdefault("params", {})["n_bits"] = int(wandb.config["attack_bits"])
                if "huffman" in wandb.config:
                    cfg.setdefault("attack", {}).setdefault("preprocessing", {})["huffman"] = bool(wandb.config["huffman"])

            # refresh opts after overrides
            opts = to_options(cfg)
    except Exception:
        pass

    # pipeline wiring
    # pipeline wiring
    bind_split_paths(cfg)
    _apply_pipeline_seeds(cfg, role="attack")

    if setting == "federated":
        from src.pipelines.federated.utils import build_exp_root

        base_save = Path(cfg["train"]["save_dir"])
        run_save_dir = build_exp_root(cfg, base_save)
        run_save_dir.mkdir(parents=True, exist_ok=True)

        cfg.setdefault("experiment", {})["run_name"] = run_save_dir.name
        cfg.setdefault("experiment", {})["run_dir"] = str(run_save_dir)
        cfg["train"]["save_dir"] = str(run_save_dir) + "/"
    else:
        run_save_dir = prepare_run_dir(cfg)

    opts = to_options(cfg)

    print("EFFECTIVE attack.name:", cfg["attack"]["name"])
    print("EFFECTIVE target_id:", cfg["experiment"].get("target_id"))
    print("EFFECTIVE seed_override:", cfg["experiment"].get("seed_override"))
    print("EFFECTIVE attack.params:", cfg.get("attack", {}).get("params", {}))

    if setting == "centralized":
        from src.pipelines.centralized.attack_runner import run_attack
    else:
        from src.pipelines.federated.attack_runner import run_attack


    res = run_attack(cfg, opts)
    save_json(run_save_dir / "results.json", res)
    save_json(run_save_dir / "config_effective.json", cfg)

    try:
        import wandb
        wandb.log(res.get("log", {}))
        wandb.log({
            "seeds/seed_big": cfg["experiment"]["seed_big"],
            "seeds/seed_attack": cfg["experiment"]["seed_attack"],
            "seeds/seed_effective_this_command": cfg["experiment"]["seed_effective_this_command"],
            "experiment/target_id": cfg["experiment"].get("target_id"),
            "experiment/seed_override": cfg["experiment"].get("seed_override"),
        })
    except Exception:
        pass

    if run:
        run.finish()

    typer.echo({"status": "done"})

@app.command()
def defend(
    config: Path = typer.Option(..., "--config", "-c"),
    target_id: Optional[str] = typer.Option(None, "--target-id"),
    case: Optional[str] = typer.Option(None, "--case"),
    defend_bits: Optional[int] = typer.Option(None, "--defend-bits"),
    seed_override: Optional[int] = typer.Option(None, "--seed-override"),
):
    cfg = load_experiment(config)

    # CLI overrides
    if target_id is not None:
        cfg.setdefault("experiment", {})["target_id"] = target_id
    if case is not None:
        cfg.setdefault("selection", {})["case"] = case
    if defend_bits is not None:
        cfg.setdefault("defense", {}).setdefault("params", {})["n_bits_defended"] = int(defend_bits)
    if seed_override is not None:
        cfg.setdefault("experiment", {})["seed_override"] = int(seed_override)

    setting = cfg["experiment"]["setting"]

    run = _wb_init(
        cfg,
        name=f"{cfg['defense']['name']}-{cfg['experiment']['setting']}",
        tags=["defense", cfg["defense"]["name"], setting],
    )

    # Apply W&B sweep overrides BEFORE binding paths / seeds
    try:
        import wandb
        if wandb.run is not None:
            if "target_id" in wandb.config:
                cfg.setdefault("experiment", {})["target_id"] = str(wandb.config["target_id"])
            if "case" in wandb.config:
                cfg.setdefault("selection", {})["case"] = str(wandb.config["case"])
            if "defend_bits" in wandb.config:
                cfg.setdefault("defense", {}).setdefault("params", {})["n_bits_defended"] = int(wandb.config["defend_bits"])
            if "seed_override" in wandb.config and wandb.config["seed_override"] is not None:
                cfg.setdefault("experiment", {})["seed_override"] = int(wandb.config["seed_override"])
    except Exception:
        pass

    bind_split_paths(cfg)                 # should set dataset paths based on target_id
    _apply_pipeline_seeds(cfg, role="defense")  # should populate experiment.seed_defense etc.

    # ---- resolve attacked checkpoint based on selection.case ----
    sel = cfg.get("selection", {})
    case_name = sel.get("case")
    cases = sel.get("cases", {})
    if not case_name or case_name not in cases:
        raise KeyError(f"selection.case='{case_name}' not found in selection.cases keys={list(cases.keys())[:10]}")

    case_cfg = cases[case_name]
    approach = case_cfg["approach"]
    huffman = bool(case_cfg["huffman"])
    attack_bits = int(case_cfg["attack_bits"])
    compressed = 1 if huffman else 0

    # set attacker metadata so downstream can decode consistently if it expects these fields
    cfg.setdefault("defense", {}).setdefault("params", {}).setdefault("attacker", {})
    cfg["defense"]["params"]["attacker"]["approach"] = approach
    cfg["defense"]["params"]["attacker"]["n_bits"] = attack_bits
    cfg["defense"]["params"]["attacker"].setdefault("preprocessing", {})["huffman"] = huffman

    # build attacked ckpt path from template fields in cfg
    atk = cfg["attacked_checkpoint"]
    run_dir = atk["run_dir_template"].format(
        attack_bits=attack_bits, approach=approach, compressed=compressed
    )
    attacked_ckpt = Path(atk["base_dir"]) / cfg["experiment"]["target_id"] / run_dir / atk["filename"]
    cfg["defense"]["params"]["attacked_ckpt"] = str(attacked_ckpt)

    # skip invalid defend_bits
    n_def = int(cfg["defense"]["params"]["n_bits_defended"])
    if n_def > attack_bits:
        res = {"status": "skipped", "reason": "defend_bits>attack_bits", "log": {"skipped": 1}}
        try:
            import wandb
            wandb.log(res["log"])
        except Exception:
            pass
        if run:
            run.finish()
        typer.echo(res)
        return

    opts = to_options(cfg)

    if setting == "centralized":
        from src.pipelines.centralized.defend_runner import run_defense
    else:
        from src.pipelines.federated.defend_runner import run_defense

    res = run_defense(cfg, opts)

    try:
        import wandb
        wandb.log(res.get("log", {}))
        wandb.log({
            "seeds/seed_big": cfg["experiment"]["seed_big"],
            "seeds/seed_defense": cfg["experiment"]["seed_defense"],
            "seeds/seed_effective_this_command": cfg["experiment"]["seed_effective_this_command"],
            "experiment/target_id": cfg["experiment"].get("target_id"),
            "experiment/seed_override": cfg["experiment"].get("seed_override"),
            "selection/case": case_name,
            "attack/attack_bits_selected": attack_bits,
            "attack/approach_selected": approach,
            "attack/huffman_selected": int(huffman),
            "defense/n_bits_defended": n_def,
        })
    except Exception:
        pass

    if run:
        run.finish()

    attacked_ckpt = Path(atk["base_dir"]) / cfg["experiment"]["target_id"] / run_dir / atk["filename"]
    cfg["defense"]["params"]["attacked_ckpt"] = str(attacked_ckpt)

    typer.echo({
        "target_id": cfg["experiment"]["target_id"],
        "case": case_name,
        "approach": approach,
        "huffman": huffman,
        "attack_bits": attack_bits,
        "resolved_attacked_ckpt": str(attacked_ckpt),
        "exists": attacked_ckpt.exists(),
    })

    typer.echo({"status": "ok"})

@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c"),
    target_id: Optional[str] = typer.Option(None, "--target_id"),
    seed_override: Optional[int] = typer.Option(None, "--seed-override"),
    max_epoch: Optional[int] = typer.Option(None, "--max_epoch"),
    resume_from: Optional[Path] = typer.Option(None, "--resume-from"),
):
    cfg = load_experiment(config)

    if target_id is not None:
        cfg.setdefault("experiment", {})["target_id"] = target_id
    if seed_override is not None:
        cfg.setdefault("experiment", {})["seed_override"] = int(seed_override)
    if max_epoch is not None:
        cfg.setdefault("train", {})["max_epoch"] = int(max_epoch)
    if resume_from is not None:
        cfg.setdefault("experiment", {})["resume_from"] = str(resume_from)
   
    opts = to_options(cfg)
    setting = cfg["experiment"]["setting"]

    sweep = cfg.get("train", {}).get("sweep")
    combos = [{}]
    if sweep:
        from itertools import product
        keys, vals = zip(*[(k, v) for k, v in sweep.items()])
        combos = [dict(zip(keys, x)) for x in product(*vals)]

    results_all = []
    import copy

    for override in combos:
        cfg_run = copy.deepcopy(cfg)
        opts_run = copy.deepcopy(opts)

        for k, v in override.items():
            if "train" in cfg_run and k in cfg_run["train"]:
                cfg_run["train"][k] = v
                opts_run[k] = v

        # build experiment name here, after all overrides are applied
        exp = cfg_run.setdefault("experiment", {})
        fed = cfg_run.setdefault("federated", {})
        train_cfg = cfg_run.setdefault("train", {})

        dataset_name = cfg_run.get("dataset", {}).get("name")

        if dataset_name is None:
            dataset_name = Path(cfg_run["dataset"]["dir"]).name
        
        base_name = f"hp_tuning_federated_{dataset_name}"

        exp["name"] = f"{base_name}_resume" if exp.get("resume_from") else base_name
        
        bind_split_paths(cfg_run)
        
        _apply_pipeline_seeds(cfg_run, role="train")
        opts_run = to_options(cfg_run)

        run = _wb_init(
            cfg_run,
            name=exp["name"],
            tags=["train", setting],
        )

        print("EFFECTIVE experiment.name:", cfg_run["experiment"]["name"])
        print("EFFECTIVE target_id:", cfg_run["experiment"].get("target_id"))
        print("EFFECTIVE max_epoch:", cfg_run["train"].get("max_epoch"))
        print("EFFECTIVE dataset.dir:", cfg_run["dataset"]["dir"])
        print("EFFECTIVE train.save_dir:", cfg_run["train"]["save_dir"])

        try:
            import wandb
            wandb.config.update({
                "effective/experiment_name": cfg_run["experiment"]["name"],
                "effective/target_id": cfg_run["experiment"].get("target_id"),
                "effective/max_epoch": cfg_run["train"].get("max_epoch"),
                "effective/dataset_dir": cfg_run["dataset"]["dir"],
                "effective/save_dir": cfg_run["train"]["save_dir"],
            }, allow_val_change=True)
        except Exception:
            pass

        if setting == "centralized":
            from src.training.train import run_benign_train
            res = run_benign_train(cfg_run, opts_run)
        else:
            from src.pipelines.federated.benign_runner import run_benign_federated
            res = run_benign_federated(cfg_run, opts_run)

        results_all.append(res)

        try:
            import wandb
            wandb.log(res.get("log", {}))
            wandb.log({
                "seeds/seed_big": cfg_run["experiment"]["seed_big"],
                "seeds/seed": cfg_run["experiment"]["seed"],
                "seeds/seed_split": cfg_run["experiment"]["seed_split"],
                "seeds/seed_train": cfg_run["experiment"]["seed_train"],
                "seeds/seed_effective_this_command": cfg_run["experiment"]["seed_effective_this_command"],
                "experiment/name": cfg_run["experiment"]["name"],
                "experiment/target_id": cfg_run["experiment"].get("target_id"),
                "experiment/seed_override": cfg_run["experiment"].get("seed_override"),
                "train/max_epoch": cfg_run["train"].get("max_epoch"),
            })
        except Exception:
            pass

        if run:
            run.finish()

    typer.echo({"runs": len(results_all)})

@app.command()
def evaluate(config: Path = typer.Option(..., "--config", "-c")):
    cfg = load_experiment(config)
    bind_split_paths(cfg)
    opts = to_options(cfg)
    print("EFFECTIVE target_id:", cfg["experiment"]["target_id"])
    print("EFFECTIVE dataset.data_dir:", cfg["dataset"]["data_dir"])
    print("EFFECTIVE predict.model_path:", cfg["predict"]["model_path"])

    from src.training.predict import evaluate_benign

    P, R, F1, FP, FN, elapsed_time = evaluate_benign(cfg, opts)

    metrics = {
        "utility/precision": float(P),
        "utility/recall": float(R),
        "utility/f1": float(F1),
        "utility/fp": int(FP),
        "utility/fn": int(FN),
        "utility/elapsed_s": float(elapsed_time),
    }

    try:
        import wandb
        exp = cfg.get("experiment", {})
        wandb.init(
            project=exp.get("project", exp.get("name", "white-box-attack")),
            entity=exp.get("entity", None),
            name=f'{exp.get("name","eval")}/benign/{exp.get("target_id","na")}',
            group=exp.get("name", None),
            job_type="eval_benign",
            config=cfg,
            reinit=True,
        )
        wandb.log(metrics)
        wandb.finish()
    except Exception:
        pass

    typer.echo({"status": "evaluated", **metrics})


@app.command()
def evaluate_federated_rounds(
    config: Path = typer.Option(..., "--config", "-c"),
    target_id: Optional[str] = typer.Option(None, "--target-id", "--target_id"),
    max_epoch: Optional[int] = typer.Option(None, "--max-epoch", "--max_epoch"),
    resume_from: Optional[Path] = typer.Option(None, "--resume-from"),
):
    import copy
    import re
    import pandas as pd
    import torch
    from pathlib import Path

    cfg = load_experiment(config)

    if target_id is not None:
        cfg.setdefault("experiment", {})["target_id"] = target_id
    if max_epoch is not None:
        cfg.setdefault("train", {})["max_epoch"] = int(max_epoch)
    if resume_from is not None:
        cfg.setdefault("experiment", {})["resume_from"] = str(resume_from)

    exp = cfg.setdefault("experiment", {})
    fed = cfg.setdefault("federated", {})
    train_cfg = cfg.setdefault("train", {})

    base_name = (
        f"hp_tuning_federated_hdfs_"
        f"{exp.get('target_id', 'nosplit')}_"
        f"{fed.get('rounds', 'norounds')}fr_"
        f"{train_cfg.get('max_epoch', 'noepoch')}le"
    )

    if exp.get("resume_from") is not None:
        exp["name"] = f"{base_name}_resume"
    else:
        exp["name"] = base_name

    bind_split_paths(cfg)
    opts = to_options(cfg)

    from src.training.predict import evaluate_benign

    global_dir = Path(opts["save_dir"]) / cfg["experiment"]["name"] / "global"
    if not global_dir.exists():
        raise FileNotFoundError(f"Global checkpoint dir not found: {global_dir}")

    ckpts = sorted(
        global_dir.glob(f"{opts['model_name']}_round*_last.pth"),
        key=lambda p: int(re.search(r"_round(\d+)_last\.pth$", p.name).group(1))
    )

    if not ckpts:
        raise FileNotFoundError(f"No round checkpoints found in {global_dir}")

    rows = []

    # optional: record resume metadata if available
    resumed_from_round = None
    if resume_from is not None:
        ckpt0 = torch.load(resume_from, map_location="cpu")
        resumed_from_round = int(ckpt0.get("round", 0))

    for ckpt in ckpts:
        m = re.search(r"_round(\d+)_last\.pth$", ckpt.name)
        if m is None:
            continue
        round_id = int(m.group(1))

        cfg_eval = copy.deepcopy(cfg)
        opts_eval = copy.deepcopy(opts)
        cfg_eval.setdefault("predict", {})["model_path"] = str(ckpt)
        opts_eval["model_path"] = str(ckpt)

        P, R, F1, FP, FN, elapsed_time = evaluate_benign(cfg_eval, opts_eval)

        rows.append({
            "round": round_id,
            "precision": float(P),
            "recall": float(R),
            "f1": float(F1),
            "fp": int(FP),
            "fn": int(FN),
            "elapsed_s": float(elapsed_time),
            "split": cfg["experiment"]["target_id"],
            "local_epochs": cfg["train"]["max_epoch"],
            "experiment_name": cfg["experiment"]["name"],
            "resumed": resume_from is not None,
            "resumed_from_round": resumed_from_round,
        })

    df = pd.DataFrame(rows).sort_values("round")
    out_path = global_dir / "f1_history.csv"
    df.to_csv(out_path, index=False)

    print(df)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    app()
