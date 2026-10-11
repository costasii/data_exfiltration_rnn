# src/defenses/lsb_defense.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple
import csv
import json
import importlib
import shutil
import time
import traceback
import re
from src.models.deeplog import deeplog


from tqdm import tqdm

from src.utils.lsb_utils import *


def _extract_attacker_nbits_from_name(folder_name: str) -> Optional[int]:
    """
    Parse patterns like '...atk=lsb(n5,smart)...' or 'lsb(n12' and return 5 / 12.
    Returns None if not found.
    """
    # common patterns include "lsb(n5,smart)" or "lsb(n5)" or "n5"
    m = re.search(r"lsb\(n?(\d+)", folder_name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # fallback: look for __atk=lsb(n5,...) style
    m2 = re.search(r"n(\d+)\)", folder_name)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    # last fallback: any 'n5' token
    m3 = re.search(r"\bn(\d+)\b", folder_name)
    if m3:
        try:
            return int(m3.group(1))
        except Exception:
            return None
    return None

def run_lsb_defense(cfg: Dict[str, Any], opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    LSB sanitization defense runner that produces a report similar to your attack report,
    but with 'defense' metadata. See cfg['defense'] for options (models_root, out_root, etc).
    """
    dcfg = dict(cfg.get("defense", {}))
    models_root = Path(dcfg.get("models_root", "models/attacked"))
    out_root = Path(dcfg.get("out_root", "models/defended"))
    target_model = dcfg.get("target_model")
    Model = deeplog(
        input_size=opts['input_size'],
        hidden_size=opts['hidden_size'],
        num_layers=opts['num_layers'],
        num_keys=opts['num_classes']
    )
    # **Important**: the number of LSBs the *defense* will zero
    sanitized_n_bits = dcfg.get("n_bits_sanitized")
    if sanitized_n_bits is None:
        # try opts fallback (sometimes wandb merges into opts)
        sanitized_n_bits = opts.get("n_bits_sanitized") if isinstance(opts, dict) else None
    if sanitized_n_bits is None:
        raise ValueError("Please provide defense.n_bits_sanitized in config or opts (int).")

    sanitized_n_bits = int(sanitized_n_bits)

    pattern = str(dcfg.get("param_pattern", "")) or ""
    overwrite = bool(dcfg.get("overwrite", False))
    csv_name = str(dcfg.get("csv_name", "sanitize_results.csv"))

    if not models_root.exists():
        raise FileNotFoundError(f"models_root not found: {models_root}")

    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_model_folders = sorted([p for p in models_root.iterdir() if p.is_dir()])
    if target_model:
        model_folders = [p for p in all_model_folders if target_model in p.name]
        if not model_folders:
            raise ValueError(f"No attacked model folder matching '{target_model}' found under {models_root}")
    else:
        model_folders = all_model_folders

    results_rows = []

    for model_folder in tqdm(model_folders, desc="Attacked models"):
        attacked_name = model_folder.name
        attacked_ckpt_path = model_folder / "ckpt_attacked.pth"
        if not attacked_ckpt_path.exists():
            candidates = find_model_files_in_folder(model_folder)
            if not candidates:
                continue
            attacked_ckpt_path = candidates[0]

        # parse attacker n_bits from the attacked folder name if present, include in report
        n_bits_attacker = _extract_attacker_nbits_from_name(attacked_name)

        orig_sd = load_state_dict_from_file(attacked_ckpt_path)
        orig_weights = torch.load(attacked_ckpt_path, map_location="cpu")["state_dict"]
        slots = collect_slots(orig_weights, include_names=["weight"]) 

        def _param_filter(k: str) -> bool:
            return (pattern in k) if pattern else True

        # --- sanitize ONCE using sanitized_n_bits ---
        t0 = time.time()
        sanitized_sd = sanitize_state_dict(orig_sd, n_bits=sanitized_n_bits, param_filter=_param_filter)
        # save using the structured naming (save_defended_model mirrors save_attacked_model)
        out_model_dir, out_ckpt_path = save_defended_model(
            sanitized_sd, cfg, opts, sanitized_n_bits, attacked_name, out_root=out_root
        )
        encoding_time = time.time() - t0


        # --- 2) measure utility (evaluate defended model) ---
      #  P, R, F1, FP, FN, prediction_time = evaluate_model(Model, opts, out_ckpt_path)
        
       
        # --- 3) extract raw bits sequentially from defended model ---
        defended_weights = torch.load(out_ckpt_path, map_location="cpu")["state_dict"]
        exf_stream = extract_bits_sequential(defended_weights, slots, n_bits_attacker)

        # --- 4) decode with ECC / unpack stream ---
        nsym = 10
        recovered_plain, decoding_metrics = unpack_ecc_stream(exf_stream, nsym=nsym)


        # --- 5) deserialize & similarity checks (verify) ---
        train_source = Path(opts['data_dir']) / "hdfs_train_xu"   

        original_sequences = load_training_sequences(str(train_source))

        unique_sequences, counts = deduplicate_and_sort_sequences(original_sequences)
        data_bits, count_width, w, stats = serialize_sequences(unique_sequences, counts, opts['num_classes'])
        unique_sequences_extracted, counts_extracted, _ = deserialize_sequences(recovered_plain, count_width=0, w=w, num_classes=opts["num_classes"])

        similarity_results = verify_decoding_smart(unique_sequences_extracted, counts_extracted, unique_sequences, counts,
                                             save_path_prefix=str(out_model_dir / "eval_results"),
                                             known_events=opts['num_classes'])

        # --- 6) write bitstreams for inspection (optional) ---
        similarity, distance, compared_len = write_bitstreams(out_model_dir, data_bits, recovered_plain)

        report = {
            "defense": {
                "type": "lsb_sanitize",
                "n_bits": sanitized_n_bits,
                "param_pattern": pattern or None,
                "target_attacked_model": attacked_name,
            },
            "metrics": {
                # "utility": {
                #     "precision": P,
                #     "recall": R,
                #     "f1": F1,
                #     "false_positives": FP,
                #     "false_negatives": FN,
               # },
                "timing": {
                    "encoding_time_s": float(encoding_time),
                #    "prediction_time_s": float(prediction_time) if prediction_time is not None else None,
                },
                "similarity": {
                    "hamming_similarity": similarity,
                    "hamming_distance": distance,
                    "bits_compared": compared_len,
                    "sequence_membership": similarity_results.get("sequence_membership") if isinstance(similarity_results, dict) else None,
                    "event_membership": similarity_results.get("event_membership") if isinstance(similarity_results, dict) else None,
                    "count_matches": similarity_results.get("count_matches") if isinstance(similarity_results, dict) else None,
                    "count_distance": similarity_results.get("count_distance") if isinstance(similarity_results, dict) else None,
                    "count_coverage": similarity_results.get("count_coverage") if isinstance(similarity_results, dict) else None,
                    "subsequence_coverage": similarity_results.get("subsequence_coverage") if isinstance(similarity_results, dict) else None,
                },
                "decoding": {
                    "total_chunks": decoding_metrics.get("chunks_total") if isinstance(decoding_metrics, dict) else None,
                    "chunks_decoded_ok": decoding_metrics.get("chunks_decoded_ok") if isinstance(decoding_metrics, dict) else None,
                    "decode_success_rate": decoding_metrics.get("decode_success_rate") if isinstance(decoding_metrics, dict) else None,
                    "useful_bits_intended": decoding_metrics.get("useful_bits_intended") if isinstance(decoding_metrics, dict) else None,
                    "useful_bits_decoded": decoding_metrics.get("useful_bits_decoded") if isinstance(decoding_metrics, dict) else None,
                    "dataset_coverage_decoded": decoding_metrics.get("dataset_coverage_decoded") if isinstance(decoding_metrics, dict) else None,
                    "raw_bits_used": decoding_metrics.get("raw_bits_used") if isinstance(decoding_metrics, dict) else None,
                    "enc_bits_total": decoding_metrics.get("enc_bits_total") if isinstance(decoding_metrics, dict) else None,
                    "header_bits_total": decoding_metrics.get("header_bits_total") if isinstance(decoding_metrics, dict) else None,
                    "overhead_bits_total": decoding_metrics.get("protocol_overhead_bits_total") if isinstance(decoding_metrics, dict) else None,
                    "overhead_ratio": decoding_metrics.get("overhead_ratio") if isinstance(decoding_metrics, dict) else None,
                    "per_chunk": decoding_metrics.get("per_chunk") if isinstance(decoding_metrics, dict) else None,
                    "stopped_on_incomplete_tail": decoding_metrics.get("stopped_on_incomplete_tail") if isinstance(decoding_metrics, dict) else None,
                },
            },
            "paths": {
                "attacked_ckpt": str(attacked_ckpt_path),
                "defended_ckpt": str(out_ckpt_path),
                "report_dir": str(out_model_dir),
            },
        }

    with open(out_model_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- 10) (Optional) Log to W&B if active, same keys as your original ---
    try:
        import wandb
        wandb.init()
        if wandb.run is not None:
            wandb.log({
              # defense info instead of attack
                "defense/n_bits": sanitized_n_bits,
                "defense/param_pattern": pattern or "",

                # utility
                # "utility/precision": float(P) if P is not None else None,
                # "utility/recall": float(R) if R is not None else None,
                # "utility/f1": float(F1) if F1 is not None else None,
                # "utility/false_positives": int(FP) if FP is not None else None,
                # "utility/false_negatives": int(FN) if FN is not None else None,

                # timing
                "timing/encoding_time_s": float(encoding_time),
            #    "timing/prediction_time_s": float(prediction_time) if prediction_time is not None else None,

                # similarity
                "similarity/hamming_similarity": float(similarity) if similarity is not None else None,
                "similarity/hamming_distance": int(distance) if distance is not None else None,
                "similarity/bits_compared": int(compared_len) if compared_len is not None else None,
                "similarity/sequence_membership": similarity_results.get("sequence_membership") if isinstance(similarity_results, dict) else None,
                "similarity/event_membership": similarity_results.get("event_membership") if isinstance(similarity_results, dict) else None,
                "similarity/count_matches": similarity_results.get("count_matches") if isinstance(similarity_results, dict) else None,
                "similarity/count_distance": similarity_results.get("count_distance") if isinstance(similarity_results, dict) else None,
                "similarity/count_coverage": similarity_results.get("count_coverage") if isinstance(similarity_results, dict) else None,
                "similarity/subsequence_coverage": similarity_results.get("subsequence_coverage") if isinstance(similarity_results, dict) else None,

                # decoding
                "decoding/total_chunks": decoding_metrics.get("chunks_total") if isinstance(decoding_metrics, dict) else None,
                "decoding/chunks_decoded_ok": decoding_metrics.get("chunks_decoded_ok") if isinstance(decoding_metrics, dict) else None,
                "decoding/decode_success_rate": decoding_metrics.get("decode_success_rate") if isinstance(decoding_metrics, dict) else None,
                "decoding/useful_bits_intended": decoding_metrics.get("useful_bits_intended") if isinstance(decoding_metrics, dict) else None,
                "decoding/useful_bits_decoded": decoding_metrics.get("useful_bits_decoded") if isinstance(decoding_metrics, dict) else None,
                "decoding/dataset_coverage_decoded": decoding_metrics.get("dataset_coverage_decoded") if isinstance(decoding_metrics, dict) else None,
                "decoding/raw_bits_used": decoding_metrics.get("raw_bits_used") if isinstance(decoding_metrics, dict) else None,
                "decoding/enc_bits_total": decoding_metrics.get("enc_bits_total") if isinstance(decoding_metrics, dict) else None,
                "decoding/header_bits_total": decoding_metrics.get("header_bits_total") if isinstance(decoding_metrics, dict) else None,
                "decoding/overhead_bits_total": decoding_metrics.get("protocol_overhead_bits_total") if isinstance(decoding_metrics, dict) else None,
                "decoding/overhead_ratio": decoding_metrics.get("overhead_ratio") if isinstance(decoding_metrics, dict) else None,
                            })

    except Exception:
        pass

    return report
