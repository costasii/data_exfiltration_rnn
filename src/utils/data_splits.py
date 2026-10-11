import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

import csv
import sys

csv.field_size_limit(sys.maxsize)

# ----------------------------
# Minimal config loader
# ----------------------------
def _load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def load_experiment(cfg_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    inc = cfg.get("include", []) or []
    if isinstance(inc, (str, Path)):
        inc = [inc]

    base: Dict[str, Any] = {}
    for rel in inc:
        base = _deep_merge(base, _load_yaml(cfg_path.parent / Path(rel)))

    merged = _deep_merge(base, {k: v for k, v in cfg.items() if k != "include"})
    return merged


# ----------------------------
# Common helpers
# ----------------------------
def read_lines(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8") as f:
        return f.readlines()


def write_lines(p: Path, lines: List[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def split_lines(lines: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    idx = list(range(len(lines)))
    rng.shuffle(idx)

    n_total = len(lines)
    n_train = int(n_total * train_ratio)
    if n_total > 0:
        n_train = max(1, n_train)

    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    train = [lines[i] for i in train_idx]
    test = [lines[i] for i in test_idx]
    return train, test


from typing import List

def get_three_seeds(cfg) -> List[int]:
    seeds_cfg = cfg.get("seeds", None)
    if not isinstance(seeds_cfg, dict):
        raise ValueError(
            "Expected seeds.yaml to define a mapping like:\n"
            "seeds:\n  split0: 123\n  split1: 456\n  split2: 789"
        )

    required = ["split0", "split1", "split2"]
    missing = [k for k in required if k not in seeds_cfg]
    if missing:
        raise ValueError(f"Missing seed entries in seeds.yaml: {missing}. Found: {list(seeds_cfg.keys())}")

    # Ensure correct order: split0, split1, split2
    return [int(seeds_cfg["split0"]), int(seeds_cfg["split1"]), int(seeds_cfg["split2"])]


# ----------------------------
# HDFS split (merge then split)
# ----------------------------
def make_hdfs_split(
    base_dir: Path,
    out_dir: Path,
    seed: int,
    train_ratio: float,
    train_file: str = "hdfs_train_xu",
    test_normal_file: str = "hdfs_test_normal_xu",
    test_abnormal_file: str = "hdfs_test_abnormal_xu",
) -> None:
    train_path = base_dir / train_file
    test_normal_path = base_dir / test_normal_file
    test_abnormal_path = base_dir / test_abnormal_file

    assert train_path.exists(), f"Missing: {train_path}"
    assert test_normal_path.exists(), f"Missing: {test_normal_path}"
    assert test_abnormal_path.exists(), f"Missing: {test_abnormal_path}"

    merged_lines = read_lines(train_path) + read_lines(test_normal_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    new_train, new_test_normal = split_lines(merged_lines, train_ratio=train_ratio, seed=seed)

    write_lines(out_dir / train_file, new_train)
    write_lines(out_dir / test_normal_file, new_test_normal)

    # abnormal unchanged
    shutil.copy2(test_abnormal_path, out_dir / test_abnormal_file)

    print(
        f"[HDFS {out_dir.name}] seed={seed} | total_normal={len(merged_lines)} "
        f"| train={len(new_train)} ({len(new_train)/max(1,len(merged_lines)):.4%}) "
        f"| test_normal={len(new_test_normal)} ({len(new_test_normal)/max(1,len(merged_lines)):.4%}) "
        f"| test_abnormal=copied"
    )

import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict

csv.field_size_limit(sys.maxsize)


def bgl_csv_to_seqs(csv_path: Path) -> List[List[int]]:
    """
    Reads BGL CSV with columns: ID,Event_seq
    Returns list of sequences as lists of ints.
    """
    assert csv_path.exists(), f"Missing: {csv_path}"
    seqs: List[List[int]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "Event_seq" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must have 'Event_seq' column, got {reader.fieldnames}")

        for row in reader:
            seq = (row.get("Event_seq") or "").strip()
            if not seq:
                continue
            tokens = seq.split()
            seqs.append([(int(t) - 1) for t in tokens])

    return seqs


def build_dense_mapping(seqs_a: List[List[int]], seqs_b: List[List[int]]) -> Dict[int, int]:
    """
    Build a deterministic 1-to-1 mapping old_id -> new_id in [1..K]
    using all unique IDs from both lists.
    """
    uniq = set()
    for seqs in (seqs_a, seqs_b):
        for s in seqs:
            uniq.update(s)

    sorted_ids = sorted(uniq)
    return {old: new for new, old in enumerate(sorted_ids)}


def apply_mapping(seqs: List[List[int]], mapping: Dict[int, int]) -> List[str]:
    """
    Apply mapping and return list of lines (HDFS-style) with trailing '\n'
    """
    out_lines: List[str] = []
    for s in seqs:
        remapped = [str(mapping[x]) for x in s]
        out_lines.append(" ".join(remapped) + "\n")
    return out_lines


def make_bgl_split(
    raw_dir: Path,
    out_dir: Path,
    seed: int,
    train_ratio: float,
    normal_csv: str = "normal.csv",
    abnormal_csv: str = "abnormal.csv",
    out_train: str = "bgl_train",
    out_test_normal: str = "bgl_test_normal",
    out_test_abnormal: str = "bgl_test_abnormal",
) -> None:
    normal_path = raw_dir / normal_csv
    abnormal_path = raw_dir / abnormal_csv

    # 1) read as int sequences
    normal_seqs = bgl_csv_to_seqs(normal_path)
    abnormal_seqs = bgl_csv_to_seqs(abnormal_path)

    # 2) build dense mapping across BOTH normal + abnormal
    mapping = build_dense_mapping(normal_seqs, abnormal_seqs)
    K = len(mapping)

    # 3) apply mapping to both
    normal_lines = apply_mapping(normal_seqs, mapping)
    abnormal_lines = apply_mapping(abnormal_seqs, mapping)

    # 4) split mapped normal
    out_dir.mkdir(parents=True, exist_ok=True)
    new_train, new_test_normal = split_lines(normal_lines, train_ratio=train_ratio, seed=seed)

    # 5) write files
    write_lines(out_dir / out_train, new_train)
    write_lines(out_dir / out_test_normal, new_test_normal)
    write_lines(out_dir / out_test_abnormal, abnormal_lines)

    # sanity stats
    # (IDs must be 0..K-1)
    print(
        f"[BGL {out_dir.name}] seed={seed} | K(unique_events)={K} "
        f"| total_normal={len(normal_lines)} train={len(new_train)} "
        f"| test_normal={len(new_test_normal)} abnormal={len(abnormal_lines)} "
        f"| id_range_new=1..{K}"
    )

# ----------------------------
# Main entry
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default=None, help="override dataset name: hdfs or bgl")
    args = parser.parse_args()

    cfg = load_experiment(args.config)

    ds_name = (args.dataset or cfg.get("dataset", {}).get("name") or "").lower()
    if ds_name not in {"hdfs", "bgl"}:
        raise ValueError(f"dataset.name must be 'hdfs' or 'bgl' (got: {ds_name!r})")

    seeds = get_three_seeds(cfg)
    if len(seeds) != 3:
        raise ValueError(f"Need exactly 3 seeds for your 3 big experiments, got: {seeds}")

    train_ratio = float(cfg.get("dataset", {}).get("train_ratio", 0.01))

    if ds_name == "hdfs":
        # base_dir contains split0 already, at base_dir/split0
        base_dir = Path(cfg.get("dataset", {}).get("base_dir", "data/processed/hdfs_xu"))
        out_base = base_dir  # split folders live here: base_dir/split{0,1,2}

        split0_dir = out_base / "split0"
        if not split0_dir.exists():
            raise FileNotFoundError(
                f"Expected existing HDFS split0 at {split0_dir}. "
                f"(You said split0 already exists.)"
            )

        # split0 already exists -> only create split1 and split2 derived from split0
        for split_id, seed in enumerate(seeds):
            out_dir = out_base / f"split{split_id}"

            if split_id == 0:
                # do NOT regenerate split0
                continue

            # derive new splits from the existing split0
            make_hdfs_split(base_dir=split0_dir, out_dir=out_dir, seed=seed, train_ratio=train_ratio)

    else:  # bgl
        raw_dir = Path(cfg.get("dataset", {}).get("raw_dir", "data/raw/bgl"))
        out_base = Path(cfg.get("dataset", {}).get("out_dir", "data/processed/bgl"))
        for split_id, seed in enumerate(seeds):
            out_dir = out_base / f"split{split_id}"
            make_bgl_split(raw_dir=raw_dir, out_dir=out_dir, seed=seed, train_ratio=train_ratio)

if __name__ == "__main__":
    main()
