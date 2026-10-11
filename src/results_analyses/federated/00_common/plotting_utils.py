
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from experiment_registry import EXPERIMENTS, BASE_DIRS, SPLITS, SAVE_DIR

# =========================
# Loading
# =========================

import json

def load_round_history(dataset, folder_name, base_dirs=None):
    if base_dirs is None:
        base_dirs = BASE_DIRS

    folder = base_dirs[dataset] / folder_name

    csv_path = folder / "global" / "round_history.csv"
    json_path = folder / "results.json"

    if csv_path.exists():
        return (
            pd.read_csv(csv_path)
            .sort_values("round")
            .reset_index(drop=True)
        )

    if json_path.exists():
        with open(json_path, "r") as f:
            obj = json.load(f)

        if "round_history" not in obj:
            raise KeyError(f"No round_history in {json_path}")

        return (
            pd.DataFrame(obj["round_history"])
            .sort_values("round")
            .reset_index(drop=True)
        )

    raise FileNotFoundError(
        f"No round_history.csv or results.json found in {folder}"
    )


def get_client_columns(df, metric_name):
    return [c for c in df.columns if c.startswith("client") and c.endswith(f"/{metric_name}")]


def get_column_by_suffix(df, suffix):
    return [c for c in df.columns if c.endswith(suffix)]


def load_experiment_data(exp_key, dataset, num_clients):
    exp = EXPERIMENTS[exp_key]
    data = {}

    for setting_key, setting_cfg in exp["settings"].items():
        data[setting_key] = {}

        for split in SPLITS:
            folder = setting_cfg["folders"][dataset](split, num_clients)
            df = load_round_history(dataset, folder)
            data[setting_key][split] = df

    return data

def aggregate_training(df):
    out = pd.DataFrame()
    out["round"] = df["round"]

    train_cols = get_client_columns(df, "train_loss")
    val_cols = get_client_columns(df, "val_loss")

    out["train_loss_mean"] = df[train_cols].mean(axis=1)
    out["val_loss_mean"] = df[val_cols].mean(axis=1)

    f1_cols = get_column_by_suffix(df, "utility_f1")
    out["utility_f1"] = df[f1_cols[0]] / 100.0 if f1_cols else np.nan

    return out

# =========================
# Shared plotting helpers
# =========================

COLORS = {
    "benign": "green",
    "malicious": "lightcoral",
    "acc": "orange",
}

def mean_std_over_splits(dfs, agg_fn, metrics, **kwargs):
    per_split = [agg_fn(df, **kwargs) for df in dfs]

    out = pd.DataFrame({"round": per_split[0]["round"]})

    for metric in metrics:
        vals = pd.concat([df[metric] for df in per_split], axis=1)
        out[f"{metric}_mean"] = vals.mean(axis=1)
        out[f"{metric}_std"] = vals.std(axis=1)

    return out


def find_best_round_from_validation(df):
    agg = aggregate_training(df)
    idx = agg["val_loss_mean"].idxmin()

    return {
        "round": int(agg.loc[idx, "round"]),
        "val_loss": float(agg.loc[idx, "val_loss_mean"]),
        "utility_f1": float(agg.loc[idx, "utility_f1"]),
    }


def find_best_round_from_validation_over_splits(dfs, client_id=None):
    vals = []

    for df in dfs:
        if client_id is None:
            vals.append(aggregate_training(df)[["round", "val_loss_mean"]])
        else:
            vals.append(pd.DataFrame({
                "round": df["round"],
                "val_loss_mean": df[f"client{client_id}/val_loss"]
            }))

    out = pd.DataFrame({"round": vals[0]["round"]})
    out["val_loss_mean"] = pd.concat(
        [v["val_loss_mean"] for v in vals],
        axis=1
    ).mean(axis=1)

    idx = out["val_loss_mean"].idxmin()

    return {
        "round": int(out.loc[idx, "round"]),
        "val_loss": float(out.loc[idx, "val_loss_mean"]),
    }


def mark_best_round(ax, agg, best_round, f1_col, acc_col):
    ax.axvline(
        best_round,
        color="grey",
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="Min validation loss"
    )

    row = agg.loc[agg["round"] == best_round]
    if row.empty:
        return

    y_f1 = float(row[f1_col].iloc[0])
    y_acc = float(row[acc_col].iloc[0])

    ax.scatter([best_round], [y_f1], color="red", s=35, zorder=5)
    ax.scatter([best_round], [y_acc], color="#ba6d02", s=35, zorder=5)

    ax.text(best_round - 0.3, y_f1, f"{y_f1:.2f}", fontsize=8,
            color="red", ha="right", va="center")
    ax.text(best_round + 0.3, y_acc, f"{y_acc:.2f}", fontsize=8,
            color="#ba6d02", ha="left", va="center")


def plot_line_with_band(ax, x, y, color, label, y_std=None, linestyle="-", alpha=0.2):
    ax.plot(x, y, color=color, linestyle=linestyle, label=label)

    if y_std is not None:
        ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=alpha)


def format_axis(ax, title, x):
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.grid(True, linestyle="--", alpha=0.4)


# =========================
# Aggregation
# =========================

def aggregate_attack_local(df, mode="current"):
    f1_cols = [
        c for c in df.columns
        if c.startswith("client") and f"/{mode}/f1_macro_event_similarity" in c
    ]
    acc_cols = [
        c for c in df.columns
        if c.startswith("client") and f"/{mode}/event_accuracy" in c
    ]

    utility_cols = get_column_by_suffix(df, "utility_f1")

    return pd.DataFrame({
        "round": df["round"],
        "benign_f1": df[utility_cols[0]] / 100.0 if utility_cols else np.nan,
        "malicious_f1_mean": df[f1_cols].mean(axis=1),
        "malicious_f1_std": df[f1_cols].std(axis=1),
        "event_acc_mean": df[acc_cols].mean(axis=1),
        "event_acc_std": df[acc_cols].std(axis=1),
    })


def aggregate_attack_global_merged(df, mode="current"):
    utility_cols = get_column_by_suffix(df, "utility_f1")

    return pd.DataFrame({
        "round": df["round"],
        "benign_f1": df[utility_cols[0]] / 100.0 if utility_cols else np.nan,
        "malicious_f1": df[f"global/merged/{mode}/f1_macro_event_similarity"],
        "event_acc": df[f"global/merged/{mode}/event_accuracy"],
    })


def aggregate_attack_local_single_client(df, client_id, mode="current"):
    prefix = f"client{client_id}"

    return pd.DataFrame({
        "round": df["round"],
        "benign_f1": df["global/utility_f1"] / 100.0,
        "malicious_f1": df[f"{prefix}/{mode}/f1_macro_event_similarity"],
        "event_acc": df[f"{prefix}/{mode}/event_accuracy"],
    })


# =========================
# Drawing functions
# =========================

def draw_local_single_split(ax, df, title, cumulative=False, mode="current", show_legend=False, show_cumulative_legend=False):
    agg = aggregate_attack_local(df, mode=mode)
    x = agg["round"]
    best = find_best_round_from_validation(df)

    plot_line_with_band(ax, x, agg["benign_f1"], COLORS["benign"], "Target model F1")
    plot_line_with_band(
        ax, x,
        agg["malicious_f1_mean"],
        COLORS["malicious"],
        "Token macro F1 similarity",
        agg["malicious_f1_std"]
    )
    plot_line_with_band(
        ax, x,
        agg["event_acc_mean"],
        COLORS["acc"],
        "Token reconstruction rate",
        agg["event_acc_std"]
    )

    if cumulative:
        agg_cum = aggregate_attack_local(df, mode="cumulative")

        plot_line_with_band(
            ax, x,
            agg_cum["malicious_f1_mean"],
            COLORS["malicious"],
            "Token macro F1 similarity cumulative",
            agg_cum["malicious_f1_std"],
            linestyle="--",
            alpha=0.12
        )
        plot_line_with_band(
            ax, x,
            agg_cum["event_acc_mean"],
            COLORS["acc"],
            "Token reconstruction rate cumulative",
            agg_cum["event_acc_std"],
            linestyle="--",
            alpha=0.12
        )

    mark_best_round(
        ax,
        agg,
        best["round"],
        f1_col="malicious_f1_mean",
        acc_col="event_acc_mean"
    )

    format_axis(ax, title, x)

    if show_legend:
        ax.legend(loc="best")


def draw_over_splits(
    ax,
    dfs,
    title,
    agg_fn,
    best_client_id=None,
    cumulative=False,
    mode="current",
    show_legend=False,
    show_cumulative_legend=False,
    **agg_kwargs
):
    metrics = ["benign_f1", "malicious_f1", "event_acc"]

    agg = mean_std_over_splits(
        dfs,
        agg_fn,
        metrics,
        mode=mode,
        **agg_kwargs
    )

    x = agg["round"]
    best = find_best_round_from_validation_over_splits(
        dfs,
        client_id=best_client_id
    )

    plot_line_with_band(
        ax, x,
        agg["benign_f1_mean"],
        COLORS["benign"],
        "Target model F1",
        agg["benign_f1_std"],
        alpha=0.15
    )
    plot_line_with_band(
        ax, x,
        agg["malicious_f1_mean"],
        COLORS["malicious"],
        "Token macro F1 similarity",
        agg["malicious_f1_std"]
    )
    plot_line_with_band(
        ax, x,
        agg["event_acc_mean"],
        COLORS["acc"],
        "Token reconstruction rate",
        agg["event_acc_std"]
    )

    if cumulative:
        agg_cum = mean_std_over_splits(
            dfs,
            agg_fn,
            metrics,
            mode="cumulative",
            **agg_kwargs
        )

        plot_line_with_band(
            ax, x,
            agg_cum["malicious_f1_mean"],
            COLORS["malicious"],
            "Token macro F1 similarity cumulative",
            agg_cum["malicious_f1_std"],
            linestyle="--",
            alpha=0.12
        )
        plot_line_with_band(
            ax, x,
            agg_cum["event_acc_mean"],
            COLORS["acc"],
            "Token reconstruction rate cumulative",
            agg_cum["event_acc_std"],
            linestyle="--",
            alpha=0.12
        )
        if show_cumulative_legend:
            handles, labels = ax.get_legend_handles_labels()

            cumulative_items = [
                (h, l) for h, l in zip(handles, labels)
                if "Cumulative" in l or "cumulative" in l
            ]

            if cumulative_items:
                h, l = zip(*cumulative_items)
                ax.legend(h, l, loc="best")

    mark_best_round(
        ax,
        agg,
        best["round"],
        f1_col="malicious_f1_mean",
        acc_col="event_acc_mean"
    )

    format_axis(ax, title, x)

    if show_legend:
        ax.legend(loc="best")

# =========================
# Main plotting function
# =========================

PLOT_CONFIGS = {
    ("payload_encoding_strategy", "local"): {
        "settings_order": ["same_data", "new_data"],
        "nrows": lambda num_clients: len(SPLITS),
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.5 * len(SPLITS)),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_payload_encoding_strategy_local_c{num_clients}.png",
    },
    ("payload_encoding_strategy", "global"): {
        "settings_order": ["same_data", "new_data"],
        "nrows": lambda num_clients: 1,
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.8),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_payload_encoding_strategy_global_c{num_clients}.png",
    },
    ("payload_encoding_strategy", "local_per_client"): {
        "settings_order": ["same_data", "new_data"],
        "nrows": lambda num_clients: num_clients,
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.2 * num_clients),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_payload_encoding_strategy_local_per_client_c{num_clients}.png",
    },

    ("payload_encoding_strategy", "per_round_capacity"): {
        # Ideally include only the dynamic setting here.
        "settings_order": ["new_data"],
        "nrows": lambda num_clients: 2,
        "ncols": lambda num_clients: 1,
        "figsize": lambda num_clients: (7, 4 * num_clients),
        "filename": lambda dataset, num_clients: (
            f"{dataset}_dynamic_per_round_capacity.png"
        ),
    },

    ("temporal_dynamics", "global"): {
        "settings_order": [0, 5],
        "nrows": lambda num_clients: 1,
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.2),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_temporal_dynamics_global_c{num_clients}.png",
    },
    ("temporal_dynamics", "local_per_client"): {
        "settings_order": [0, 5],
        "nrows": lambda num_clients: num_clients,
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.2 * num_clients),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_temporal_dynamics_local_per_client_c{num_clients}.png",
    },
    ("num_clients", "local"): {
        "settings_order": [2, 10, 50],
        "nrows": lambda num_clients: len(SPLITS),
        "ncols": lambda num_clients: 3,
        "figsize": lambda num_clients: (18, 4.2 * len(SPLITS)),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_num_clients_local.png",
    },
    ("num_clients", "global"): {
        "settings_order": [2, 10, 50],
        "nrows": lambda num_clients: 1,
        "ncols": lambda num_clients: 3,
        "figsize": lambda num_clients: (18, 4.2),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_num_clients_global.png",
    },
    ("local_epochs", "local"): {
        "settings_order": [10, 100],
        "nrows": lambda num_clients: len(SPLITS),
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.2 * len(SPLITS)),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_local_epochs_local.png",
    },
    ("local_epochs", "local_per_client"): {
        "settings_order": [10, 100],
        "nrows": lambda num_clients: num_clients,
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.2 * num_clients),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_local_epochs_local_per_client.png",
    },
    ("local_epochs", "global"): {
        "settings_order": [10, 100],
        "nrows": lambda num_clients: 1,
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.2),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_local_epochs_global.png",
    },
    ("num_repetitions", "global"): {
        "settings_order": [200, 500],
        "nrows": lambda num_clients: 1,
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.2),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_num_repetitions_global.png",
    },
    ("num_repetitions", "local"): {
        "settings_order": [200, 500],
        "nrows": lambda num_clients: len(SPLITS),
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.1 * len(SPLITS)),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_num_repetitions_local.png",
    },
    ("num_repetitions", "local_per_client"): {
        "settings_order": [200, 500],
        "nrows": lambda num_clients: num_clients,
        "ncols": lambda num_clients: 2,
        "figsize": lambda num_clients: (14, 4.1 * num_clients),
        "filename": lambda dataset, num_clients: f"{dataset.lower()}_num_repetitions_local_per_client.png",
    },
}

def plot_experiment(dataset, exp_key, view, num_clients=2, mode="current"):
    cfg_plot = PLOT_CONFIGS[(exp_key, view)]
    exp = EXPERIMENTS[exp_key]
    settings_order = cfg_plot["settings_order"]

    fig, axes = plt.subplots(
        nrows=cfg_plot["nrows"](num_clients),
        ncols=cfg_plot["ncols"](num_clients),
        figsize=cfg_plot["figsize"](num_clients),
        squeeze=False
    )

    is_payload_encoding_strategy = exp_key == "payload_encoding_strategy"

    for j, setting_key in enumerate(settings_order):
        cfg = exp["settings"][setting_key]
        current_num_clients = cfg.get("num_clients", num_clients)

        data = load_experiment_data(
            exp_key=exp_key,
            dataset=dataset,
            num_clients=current_num_clients
        )

        cumulative = cfg.get("cumulative", False)

        show_cumulative_legend = (
            is_payload_encoding_strategy
            and cumulative
            and j == len(settings_order) - 1
        )

        if view == "local":
            for i, split in enumerate(SPLITS):
                df = data[setting_key][split]

                draw_local_single_split(
                    ax=axes[i, j],
                    df=df,
                    title=f"{split} | {cfg['label']}",
                    cumulative=cumulative,
                    mode=mode,
                    show_legend=(i == 0 and j == 0),
                    show_cumulative_legend=(
                        i == 0 and show_cumulative_legend
                    )
                )

        elif view == "global":
            dfs = [
                data[setting_key][split]
                for split in SPLITS
            ]

            draw_over_splits(
                ax=axes[0, j],
                dfs=dfs,
                title=cfg["label"],
                agg_fn=aggregate_attack_global_merged,
                cumulative=cumulative,
                mode=mode,
                show_legend=(j == 0),
                show_cumulative_legend=show_cumulative_legend
            )

        elif view == "local_per_client":
            dfs = [
                data[setting_key][split]
                for split in SPLITS
            ]

            for i, client_id in enumerate(
                range(current_num_clients)
            ):
                draw_over_splits(
                    ax=axes[i, j],
                    dfs=dfs,
                    title=(
                        f"Client {client_id + 1} | "
                        f"{cfg['label']}"
                    ),
                    agg_fn=aggregate_attack_local_single_client,
                    best_client_id=client_id,
                    cumulative=cumulative,
                    mode=mode,
                    show_legend=(i == 0 and j == 0),
                    show_cumulative_legend=(
                        i == 0 and show_cumulative_legend
                    ),
                    client_id=client_id
                )

        elif view == "per_round_capacity":
            dfs = [
                data[setting_key][split]
                for split in SPLITS
            ]

            for i, client_id in enumerate(range(current_num_clients)):
                draw_per_round_capacity_plot(
                    ax=axes[i, j],
                    dfs=dfs,
                    title=f"Client {client_id + 1} | {cfg['label']}",
                    agg_fn=aggregate_attack_local_single_client,
                    tokens_per_round=50,
                    rounds_are_zero_indexed=True,
                    show_legend=(i == 0 and j == 0),
                    client_id=client_id,
                    mode=mode,
                )

    for ax in axes[-1, :]:
        ax.set_xlabel("Federated round")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = SAVE_DIR / cfg_plot["filename"](
        dataset,
        num_clients
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()


from pathlib import Path
import numpy as np
import pandas as pd

BENIGN_BASE = Path("/home/mirta/dev/data_exfiltration/models/federated/parameter_tuning")


def benign_bgl_folder(split, num_clients=2):
    return (
        BENIGN_BASE
        / "bgl"
        / f"hp_tuning_federated_bgl_{split}_le30_c{num_clients}"
    )

def benign_hdfs_folders(split, num_clients=2):
    return (
        BENIGN_BASE
        / "hdfs"
        / f"hp_tuning_federated_hdfs_{split}_le10_c{num_clients}"
    )

# =========================
# Helpers
# =========================

def mean_std(vals):
    vals = np.array(vals, dtype=float)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def pm(mean, std, digits=2):
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}$"


def utility_f1_col(df):
    cols = [c for c in df.columns if c.endswith("utility_f1")]
    if not cols:
        raise KeyError("No utility_f1 column found.")
    return cols[0]


def client_val_loss_cols(df):
    return [
        c for c in df.columns
        if c.startswith("client") and c.endswith("/val_loss")
    ]


def best_round_idx(df):
    val_cols = client_val_loss_cols(df)
    if not val_cols:
        raise KeyError("No client*/val_loss columns found.")

    val_mean = df[val_cols].mean(axis=1)
    return val_mean.idxmin()


def attacked_f1_at_idx(df, idx):
    f1_col = utility_f1_col(df)
    f1 = float(df.loc[idx, f1_col])
    return f1 / 100.0 if f1 > 1 else f1


def local_client_metric_cols(df, metric, mode="current"):
    return [
        c for c in df.columns
        if c.startswith("client")
        and f"/{mode}/{metric}" in c
        and not c.startswith("global/")
    ]


def client_values_at_round(df, idx, metric, mode="current"):
    cols = local_client_metric_cols(df, metric=metric, mode=mode)

    if not cols:
        raise KeyError(f"No local client columns found for metric={metric}")

    return df.loc[idx, cols].astype(float).values


# =========================
# Benign F1 baseline
# selected by min validation loss
# =========================

def last_valid_loss(valid_log_path):
    df = pd.read_csv(valid_log_path)
    loss_cols = [c for c in df.columns if "loss" in c.lower()]

    if not loss_cols:
        raise KeyError(f"No loss column found in {valid_log_path}")

    return float(df[loss_cols[0]].iloc[-1])


def round_val_loss_from_clients(folder, round_num, num_clients):
    vals = []

    for client_id in range(0, num_clients):
        path = (
            Path(folder)
            / "federated_clients"
            / f"round_{round_num}"
            / f"client_{client_id}"
            / "valid_log.csv"
        )

        if path.exists():
            vals.append(last_valid_loss(path))

    return np.nan if not vals else float(np.nanmean(vals))


def read_f1_history(folder):
    folder = Path(folder)

    f1_path = folder / "global" / "f1_history.csv"
    round_path = folder / "global" / "round_history.csv"

    if f1_path.exists():
        df = pd.read_csv(f1_path)
        if "f1" not in df.columns:
            raise KeyError(f"No 'f1' column in {f1_path}")

        return df.rename(columns={"f1": "utility_f1"})

    if round_path.exists():
        df = pd.read_csv(round_path)
        f1_col = utility_f1_col(df)
        return df.rename(columns={f1_col: "utility_f1"})

    raise FileNotFoundError(
        f"No f1_history.csv or round_history.csv found in {folder}"
    )


def best_benign_f1_from_folder(folder, num_clients):
    folder = Path(folder)
    hist = read_f1_history(folder).sort_values("round").copy()

    hist["val_loss_mean"] = [
        round_val_loss_from_clients(folder, int(r), num_clients)
        for r in hist["round"]
    ]

    hist = hist.dropna(subset=["val_loss_mean"])

    if hist.empty:
        raise ValueError(f"No valid validation losses found in {folder}")

    best = hist.sort_values("val_loss_mean").iloc[0]

    f1 = float(best["utility_f1"])
    return f1


def benign_folder(dataset, split, num_clients):
    dataset_key = dataset.upper()

    if dataset_key == "BGL":
        return benign_bgl_folder(split, num_clients)

    if dataset_key == "HDFS":
        return benign_hdfs_folders(split, num_clients)

    raise ValueError(dataset)


# =========================
# Training time
# =========================

def round_training_time_cols(df):
    return [
        c for c in df.columns
        if c.startswith("client")
        and (
            c.endswith("/train_time")
            or c.endswith("/train_elapsed_s")
            or c.endswith("/client/train_elapsed_s")
        )
    ]


def training_time_until_round_from_history_df(df, round_num):
    df_until = df[df["round"] <= round_num].copy()

    if df_until.empty:
        raise ValueError(f"No rows found up to round {round_num}")

    if "round/train_elapsed_max_s" in df_until.columns:
        return float(df_until["round/train_elapsed_max_s"].sum())

    time_cols = round_training_time_cols(df_until)

    if not time_cols:
        raise KeyError("No training-time columns found.")

    return float(df_until[time_cols].max(axis=1).sum())


def load_round_history_from_path(folder):
    folder = Path(folder)

    csv_path = folder / "global" / "round_history.csv"
    json_path = folder / "results.json"

    if csv_path.exists():
        return pd.read_csv(csv_path).sort_values("round").reset_index(drop=True)

    if json_path.exists():
        with open(json_path, "r") as f:
            obj = json.load(f)

        if "round_history" not in obj:
            raise KeyError(f"No round_history in {json_path}")

        return (
            pd.DataFrame(obj["round_history"])
            .sort_values("round")
            .reset_index(drop=True)
        )

    raise FileNotFoundError(
        f"No round_history.csv or results.json found in {folder}"
    )


def training_time_until_round_from_path(folder, round_num):
    hist = load_round_history_from_path(folder)
    return training_time_until_round_from_history_df(hist, round_num)

def training_time_until_round(dataset, folder_name, round_num):
    hist = load_round_history(dataset, folder_name)

    return training_time_until_round_from_history_df(
        hist,
        round_num
    )
def get_total_train_tokens(dataset, split):
    """
    Counts total number of event tokens in the train set.

    Expected files:
    bgl      -> bgl_train
    hdfs     -> hdfs_train_xu
    """

    split_dir = Path(
        f"/home/mirta/dev/data_exfiltration/data/processed/"
        f"{'hdfs_xu' if dataset == 'hdfs' or dataset == 'HDFS' else dataset.lower()}/"
        f"{split}"
    )

    if dataset == "bgl" or dataset == "BGL":
        train_file = split_dir / "bgl_train"

    elif dataset == "hdfs" or dataset == "HDFS":
        train_file = split_dir / "hdfs_train_xu"

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not train_file.exists():
        raise FileNotFoundError(f"Missing train file: {train_file}")

    total_tokens = 0

    with open(train_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            total_tokens += len(line.split())

    return total_tokens

# =========================
# One split summary
# =========================
def summarize_federated_setting_for_split(
    dataset,
    split,
    exp_key,
    setting_key,
    num_clients=2,
    mode="current",
    payload_per_client=50,
    round_selection="min_val",   # "min_val" or "best_attack"
):
    dataset_key = dataset.upper()

    cfg = EXPERIMENTS[exp_key]["settings"][setting_key]
    c = cfg.get("num_clients", num_clients)

    data = load_experiment_data(
        exp_key=exp_key,
        dataset=dataset,
        num_clients=c
    )

    df = data[setting_key][split]
    attack_folder_name = cfg["folders"][dataset_key](split, c)
    baseline_folder = benign_folder(dataset, split, c)
    min_round = min_attack_round_for_setting(setting_key, cfg)


    if round_selection == "min_val":
        idx = best_round_idx(df)

    elif round_selection == "best_attack":
        idx = best_attack_round_idx_with_f1_constraint(
            df=df,
            baseline_folder=baseline_folder,
            mode=mode,
            max_f1_drop=2.0,
            min_round=min_round,
        )
    else:
        raise ValueError(round_selection)

    selected_round = int(df.loc[idx, "round"])

    sim_vals = client_values_at_round(
        df, idx, metric="f1_macro_event_similarity", mode=mode
    )

    acc_vals = client_values_at_round(
        df, idx, metric="event_accuracy", mode=mode
    )

    

    # unchanged: benign F1 still selected as before
    benign_f1 = best_benign_f1_from_folder(
        baseline_folder,
        num_clients=c
    )

    attacked_f1 = attacked_f1_at_idx(df, idx) * 100

    # unchanged baseline logic, but evaluated until the chosen round
    baseline_time = training_time_until_round_from_path(
        baseline_folder,
        selected_round
    )

    attacked_time = training_time_until_round(
        dataset=dataset_key,
        folder_name=attack_folder_name,
        round_num=selected_round,
    )

    train_tokens = get_total_train_tokens(dataset, split)
    ec = payload_per_client * c / train_tokens * 100

    return {
        "setting": cfg["label"],
        "num_clients": c,
        "split": split,
        "round": selected_round,

        "EC": ec,

        "S_mean_clients": float(np.nanmean(sim_vals)),
        "S_std_clients": float(np.nanstd(sim_vals)),

        "EA_mean_clients": float(np.nanmean(acc_vals)),
        "EA_std_clients": float(np.nanstd(acc_vals)),

        "benign_time": baseline_time,
        "attack_time": attacked_time,
        "delta_OH": attacked_time - baseline_time,

        "benign_f1": benign_f1,
        "attack_f1": attacked_f1,
        "delta_F1": attacked_f1 - benign_f1,
    }


# =========================
# Aggregate across splits
# =========================
def round_list_str(rounds):
    return ", ".join(str(int(r)) for r in rounds)


def aggregate_split_summaries(split_df):
    rows = []

    group_cols = ["setting", "num_clients"]

    for (setting, num_clients), g in split_df.groupby(group_cols, sort=False):
        rows.append({
            "setting": setting,
            "num_clients": num_clients,

            # keep actual chosen rounds from split0/split1/split2
            "rounds": round_list_str(g["round"].values),

            "EC_mean": g["EC"].mean(),

            "S_mean": g["S_mean_clients"].mean(),
            "S_std": g["S_std_clients"].mean(),

            "EA_mean": g["EA_mean_clients"].mean(),
            "EA_std": g["EA_std_clients"].mean(),

            "benign_time_mean": g["benign_time"].mean(),
            "attack_time_mean": g["attack_time"].mean(),
            "delta_OH_mean": g["delta_OH"].mean(),
            "delta_OH_std": g["delta_OH"].std(),

            "benign_f1_mean": g["benign_f1"].mean(),
            "benign_f1_std": g["benign_f1"].std(),

            "attack_f1_mean": g["attack_f1"].mean(),
            "attack_f1_std": g["attack_f1"].std(),

            "delta_F1_mean": g["delta_F1"].mean(),
            "delta_F1_std": g["delta_F1"].std(),
        })

    return pd.DataFrame(rows)

def make_federated_attack_summary_table(
    dataset,
    experiments,
    mode="current",
    payload_per_client=50,
    round_selection="min_val",   # "min_val" or "best_attack"
):
    split_rows = []

    for split in SPLITS:
        for exp_key, setting_key, num_clients in experiments:
            split_rows.append(
                summarize_federated_setting_for_split(
                    dataset=dataset,
                    split=split,
                    exp_key=exp_key,
                    setting_key=setting_key,
                    num_clients=num_clients,
                    mode=mode,
                    payload_per_client=payload_per_client,
                    round_selection=round_selection,
                )
            )

    split_df = pd.DataFrame(split_rows)
    summary_df = aggregate_split_summaries(split_df)

    return summary_df, split_df


def federated_summary_to_latex_rows(df):
    df = df.copy()

    # remove duplicate rows only for 2-client setting
    df = df[~(
        (df["num_clients"] == 2)
        & (df["setting"].isin(["Start round = 0", "200 rep"]))
    )]

    setting_order = {
        "Static": 0,
        "Dynamic": 1,
        "Start round = 5": 2,
        "500 rep": 3,
        "default": 4,
    }

    df["setting_order"] = df["setting"].map(setting_order)

    df = df.sort_values(
        ["num_clients", "setting_order"]
    )

    rows = []

    for _, r in df.iterrows():
        rows.append(
            f"{int(r['num_clients'])} & "
            f"{r['rounds']} & "
            f"{r['setting']} & "
            f"{r['EC_mean']:.2f} & "
            f"{pm(r['S_mean'], r['S_std'])} & "
            f"{pm(r['delta_OH_mean'], r['delta_OH_std'])} & "
            f"{pm(r['delta_F1_mean'], r['delta_F1_std'])} \\\\"
        )

    return "\n".join(rows)

def save_federated_summary_tex_rows(df, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tex_rows = federated_summary_to_latex_rows(df)

    with open(out_path, "w") as f:
        f.write(tex_rows)

    print(f"Saved: {out_path}")

def inspect_best_attack_candidates(
    df,
    baseline_folder,
    mode="current",
    min_round=1,
    max_benign_drop_from_peak=15.0,
):
    sim_cols = local_client_metric_cols(
        df,
        metric="f1_macro_event_similarity",
        mode=mode
    )

    df_tmp = df.copy()
    df_tmp = df_tmp[df_tmp["round"] >= min_round].copy()

    df_tmp["attack_similarity_mean"] = df_tmp[sim_cols].mean(axis=1)

    df_tmp["benign_f1"] = df_tmp["round"].apply(
        lambda r: benign_f1_at_round(baseline_folder, int(r))
    )

    df_tmp["attacked_f1"] = [
        attacked_f1_at_idx(df_tmp, idx) * 100
        for idx in df_tmp.index
    ]

    df_tmp = df_tmp.dropna(
        subset=["attack_similarity_mean", "benign_f1", "attacked_f1"]
    )

    max_benign_f1 = df_tmp["benign_f1"].max()
    min_allowed_benign_f1 = max_benign_f1 - max_benign_drop_from_peak

    df_tmp["benign_drop_from_peak"] = max_benign_f1 - df_tmp["benign_f1"]
    df_tmp["passes_benign_constraint"] = (
        df_tmp["benign_f1"] >= min_allowed_benign_f1
    )
    df_tmp["f1_drop"] = df_tmp["benign_f1"] - df_tmp["attacked_f1"]

    return df_tmp[[
        "round",
        "attack_similarity_mean",
        "benign_f1",
        "benign_drop_from_peak",
        "passes_benign_constraint",
        "attacked_f1",
        "f1_drop",
    ]].sort_values(
        "attack_similarity_mean",
        ascending=False
    )
def best_attack_round_idx_with_f1_constraint(
    df,
    baseline_folder,
    mode="current",
    max_f1_drop=2.0,
    max_benign_drop_from_peak=15.0,
    min_round=1,
):
    sim_cols = local_client_metric_cols(
        df,
        metric="f1_macro_event_similarity",
        mode=mode
    )

    if not sim_cols:
        raise KeyError("No local similarity columns found.")

    df_tmp = df.copy()
    df_tmp = df_tmp[df_tmp["round"] >= min_round].copy()

    if df_tmp.empty:
        raise ValueError(f"No rounds available after min_round={min_round}")

    df_tmp["attack_similarity_mean"] = df_tmp[sim_cols].mean(axis=1)

    df_tmp["benign_f1"] = df_tmp["round"].apply(
        lambda r: benign_f1_at_round(baseline_folder, int(r))
    )

    df_tmp["attacked_f1"] = [
        attacked_f1_at_idx(df_tmp, idx) * 100
        for idx in df_tmp.index
    ]

    df_tmp = df_tmp.dropna(
        subset=["attack_similarity_mean", "benign_f1", "attacked_f1"]
    )

    if df_tmp.empty:
        raise ValueError("No valid rows after removing NaN metrics.")

    max_benign_f1 = df_tmp["benign_f1"].max()
    min_allowed_benign_f1 = max_benign_f1 - max_benign_drop_from_peak

    df_tmp = df_tmp[df_tmp["benign_f1"] >= min_allowed_benign_f1].copy()

    if df_tmp.empty:
        raise ValueError(
            "No rounds satisfy benign fidelity constraint: "
            f"benign_f1 >= {min_allowed_benign_f1:.2f}"
        )

    df_tmp["f1_drop"] = df_tmp["benign_f1"] - df_tmp["attacked_f1"]

    # First: highest similarity among rounds with acceptable F1 drop
    candidates = df_tmp[df_tmp["f1_drop"] <= max_f1_drop].copy()

    if not candidates.empty:
        return candidates.sort_values(
            "attack_similarity_mean",
            ascending=False
        ).index[0]

    # Fallback: smallest F1 drop; if tied, highest similarity
    fallback = df_tmp.sort_values(
        ["f1_drop", "attack_similarity_mean"],
        ascending=[True, False]
    )

    return fallback.index[0]

def benign_f1_at_round(folder, round_num):
    hist = read_f1_history(folder).sort_values("round").copy()

    row = hist[hist["round"] == round_num]

    if row.empty:
        raise ValueError(f"Round {round_num} not found in {folder}")

    f1 = float(row.iloc[0]["utility_f1"])
    return f1 

def min_attack_round_for_setting(setting_key, cfg):
    if setting_key == 5:
        return 5

    label = str(cfg.get("label", "")).lower()
    if "start round = 5" in label or "start 5" in label:
        return 5

    return 1

# for cumulative plot
def aggregate_attack_per_round_reconstruction(
    dfs,
    agg_fn,
    **agg_kwargs,
):
    prepared = []

    for split_idx, df in enumerate(dfs):
        split_df = agg_fn(df, **agg_kwargs).copy()

        required_columns = {
            "round",
            "benign_f1",
            "malicious_f1",
            "event_acc",
        }

        missing_columns = required_columns.difference(split_df.columns)

        if missing_columns:
            raise KeyError(
                f"Aggregation output for split {split_idx} is missing: "
                f"{sorted(missing_columns)}"
            )

        split_df = split_df[
            ["round", "benign_f1", "malicious_f1", "event_acc"]
        ].copy()

        split_df["split_idx"] = split_idx
        prepared.append(split_df)

    merged = pd.concat(prepared, ignore_index=True)

    result = (
        merged
        .groupby("round", as_index=False)
        .agg(
            benign_f1_mean=("benign_f1", "mean"),
            benign_f1_std=("benign_f1", "std"),
            similarity_mean=("malicious_f1", "mean"),
            similarity_std=("malicious_f1", "std"),
            reconstruction_mean=("event_acc", "mean"),
            reconstruction_std=("event_acc", "std"),
        )
        .sort_values("round")
        .reset_index(drop=True)
    )

    result[
        ["benign_f1_std", "similarity_std", "reconstruction_std"]
    ] = result[
        ["benign_f1_std", "similarity_std", "reconstruction_std"]
    ].fillna(0.0)

    return result
def draw_per_round_capacity_plot(
    ax,
    dfs,
    title,
    agg_fn,
    tokens_per_round=50,
    rounds_are_zero_indexed=True,
    min_marker_size=40,
    max_marker_size=350,
    show_legend=True,
    **agg_kwargs,
):
    """
    Plot per-round reconstruction quality across data splits.

    The aggregation function determines which model or client output is used.

    Marker position:
        Reconstruction quality for the new payload encoded in that round.

    Marker size:
        Total exfiltration capacity reached by that round.
    """
    result = aggregate_attack_per_round_reconstruction(
        dfs=dfs,
        agg_fn=agg_fn,
        **agg_kwargs,
    )

    if result.empty:
        raise ValueError("The aggregated reconstruction data is empty.")

    rounds = result["round"].to_numpy()
    ax.plot(
        rounds,
        result["benign_f1_mean"],
        color="green",
        linewidth=2,
        label="Benign F1",
        zorder=1,
    )
    ax.fill_between(
        rounds,
        result["benign_f1_mean"] - result["benign_f1_std"],
        result["benign_f1_mean"] + result["benign_f1_std"],
        color="green",
        alpha=0.15,
        zorder=0,
    )

    if rounds_are_zero_indexed:
        attack_round_numbers = rounds + 1
    else:
        attack_round_numbers = rounds

    capacities = tokens_per_round * attack_round_numbers

    if capacities.max() == capacities.min():
        marker_sizes = np.full(
            len(capacities),
            min_marker_size,
            dtype=float,
        )
    else:
        marker_sizes = min_marker_size + (
            (capacities - capacities.min())
            / (capacities.max() - capacities.min())
        ) * (max_marker_size - min_marker_size)

    similarity_color = ax._get_lines.get_next_color()
    reconstruction_color = ax._get_lines.get_next_color()

    ax.errorbar(
        rounds,
        result["similarity_mean"],
        yerr=result["similarity_std"],
        linestyle="none",
        capsize=3,
        color=similarity_color,
        alpha=0.7,
        zorder=2,
    )

    ax.scatter(
        rounds,
        result["similarity_mean"],
        s=marker_sizes,
        marker="o",
        color=similarity_color,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
        label="Macro-F1 similarity",
        zorder=3,
    )

    ax.errorbar(
        rounds,
        result["reconstruction_mean"],
        yerr=result["reconstruction_std"],
        linestyle="none",
        capsize=3,
        color=reconstruction_color,
        alpha=0.7,
        zorder=2,
    )

    ax.scatter(
        rounds,
        result["reconstruction_mean"],
        s=marker_sizes,
        marker="s",
        color=reconstruction_color,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.5,
        label="Token reconstruction rate",
        zorder=3,
    )

    ax.set_title(title)
    ax.set_xticks(rounds)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Per-round reconstruction quality")
    ax.grid(alpha=0.25)

    if show_legend:
        metric_legend = ax.legend(
            loc="lower left",
            title="Metric",
        )
        ax.add_artist(metric_legend)

        capacity_indices = sorted({
            0,
            len(capacities) // 2,
            len(capacities) - 1,
        })

        capacity_handles = [
            ax.scatter(
                [],
                [],
                s=marker_sizes[idx],
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                label=f"{int(capacities[idx])} tokens",
            )
            for idx in capacity_indices
        ]

        ax.legend(
            handles=capacity_handles,
            title="Exfiltration capacity",
            loc="lower right",
        )