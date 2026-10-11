# plot_single_client_experiments.py

from single_client_experiment_registry import (
    BASE_DIRS,
    SINGLE_CLIENT_EXPERIMENTS,
    SPLITS,
    SAVE_DIR
)
from plotting_utils import (
    load_round_history,
    draw_over_splits,
    aggregate_attack_local_single_client,
)


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_single_client_experiment_data(
    exp_key,
    dataset,
    experiments,
    splits,
    base_dirs,
):
    exp = experiments[exp_key]
    data = {}

    for setting_key, setting_cfg in exp["settings"].items():
        data[setting_key] = {}

        for split in splits:
            folder_name = setting_cfg["folders"][dataset](split)

            data[setting_key][split] = load_round_history(
                dataset=dataset,
                folder_name=folder_name,
                base_dirs=base_dirs,
            )

    return data

SINGLE_CLIENT_PLOT_CONFIGS = {
    ("num_clients", "attack_metrics"): {
        "settings_order": [5, 10, 50],
        "nrows": 1,
        "ncols": 3,
        "figsize": (18, 5),
        "filename": lambda dataset: (
            f"{dataset.lower()}_single_attacked_client_metrics.png"
        ),
    },
    ("num_clients", "per_round_capacity"): {
        "settings_order": [5, 10, 50],
        "nrows": 1,
        "ncols": 3,
        "figsize": (18, 5),
        "filename": lambda dataset: (
            f"{dataset.lower()}_single_attacked_client_capacity.png"
        ),
    },
}

def plot_single_attacked_client_experiment(
    dataset,
    exp_key="num_clients",
    view="attack_metrics",
    mode="current",
):
    plot_cfg = SINGLE_CLIENT_PLOT_CONFIGS[(exp_key, view)]
    exp = SINGLE_CLIENT_EXPERIMENTS[exp_key]
    settings_order = plot_cfg["settings_order"]

    data = load_single_client_experiment_data(
        dataset=dataset,
        exp_key=exp_key,
    )

    fig, axes = plt.subplots(
        nrows=plot_cfg["nrows"],
        ncols=plot_cfg["ncols"],
        figsize=plot_cfg["figsize"],
        squeeze=False,
    )

    for j, setting_key in enumerate(settings_order):
        cfg = exp["settings"][setting_key]

        dfs = [
            data[setting_key][split]
            for split in SPLITS
        ]

        attacked_client_id = cfg["attacked_client_id"]
        ax = axes[0, j]

        if view == "attack_metrics":
            draw_over_splits(
                ax=ax,
                dfs=dfs,
                title=cfg["label"],
                agg_fn=aggregate_attack_local_single_client,
                best_client_id=attacked_client_id,
                client_id=attacked_client_id,
                cumulative=cfg.get("cumulative", False),
                mode=mode,
                show_legend=(j == 0),
                show_cumulative_legend=False,
            )

        elif view == "per_round_capacity":
            draw_per_round_capacity_plot(
                ax=ax,
                dfs=dfs,
                title=cfg["label"],
                agg_fn=aggregate_attack_local_single_client,
                client_id=attacked_client_id,
                tokens_per_round=50,
                rounds_are_zero_indexed=True,
                mode=mode,
                show_legend=(j == 0),
            )

        ax.set_xlabel("Federated round")

    fig.suptitle(
        f"{dataset}: attack against one client",
        fontsize=15,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = (
        SAVE_DIR
        / plot_cfg["filename"](dataset)
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return save_path

def plot_client_count_comparison(
    dataset,
    mode="current",
):
    exp = SINGLE_CLIENT_EXPERIMENTS["num_clients"]
    data = load_single_client_experiment_data(
        dataset=dataset,
        exp_key="num_clients",
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for setting_key, cfg in exp["settings"].items():
        dfs = [
            data[setting_key][split]
            for split in SPLITS
        ]

        draw_over_splits(
            ax=ax,
            dfs=dfs,
            title="",
            agg_fn=aggregate_attack_local_single_client,
            best_client_id=cfg["attacked_client_id"],
            client_id=cfg["attacked_client_id"],
            cumulative=False,
            mode=mode,
            show_legend=True,
            show_cumulative_legend=False,
            label_prefix=cfg["label"],
        )

    ax.set_title(f"{dataset}: attack against one client")
    ax.set_xlabel("Federated round")

    plt.tight_layout()

    save_path = (
        SAVE_DIR_SINGLE_CLIENT
        / f"{dataset.lower()}_client_count_comparison.png"
    )

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path