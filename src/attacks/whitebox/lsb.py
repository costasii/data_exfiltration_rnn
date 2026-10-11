import torch
import struct
import time
import wandb
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd
import torch.nn as nn
from src.models.deeplog import deeplog
from src.utils.lsb_utils import *


from typing import Dict, Any

def run_lsb_attack(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    approach = str(cfg["attack"]["params"].get("approach", "baseline"))

    if approach == "baseline":
        from src.attacks.whitebox.lsb_baseline import run_lsb_baseline
        return run_lsb_baseline(cfg, options)

    elif approach == "adapted":
        from src.attacks.whitebox.lsb_adapted import run_lsb_adapted
        return run_lsb_adapted(cfg, options)
    
    elif approach == "redundant":
        from src.attacks.whitebox.lsb_redundant import run_lsb_redundant
        metrics = run_lsb_redundant(cfg, options)

    elif approach == "hamming":
        from src.attacks.whitebox.lsb_hamming import run_lsb_hamming
        return run_lsb_hamming(cfg, options)
    
    elif approach == "rs":
        from src.attacks.whitebox.lsb_rs_ecc import run_lsb_rs
        return run_lsb_rs(cfg, options)
    
    elif approach == "repetition":
        from src.attacks.whitebox.lsb_repetition import run_lsb_repetition
        return run_lsb_repetition(cfg, options)


    else:
        raise ValueError(f"Unknown LSB approach: {approach}")


# def run_lsb_attack(cfg: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Runs ONE white-box LSB attack setting (no internal loops).
#     Reads:
#       - cfg['checkpoint']['path'] (clean model .pth)
#       - cfg['attack']['params']['n_bits'], ['approach']  (naive|smart)
#     Uses:
#       - Training data
#       - Predicter for utility metrics
#     Saves:
#       - attacked checkpoint under models/attacked/<auto-name>/ckpt_attacked.pth
#       - report.json with metrics
#     Logs:
#       - If wandb.run is active, logs key metrics.
#     """
#     n_bits   = int(cfg["attack"]["params"]["n_bits"])
#     approach = str(cfg["attack"]["params"]["approach"])

        
#     # --- 1) Build model instance (same as your original) ---
#     Model = deeplog(
#         input_size=options['input_size'],
#         hidden_size=options['hidden_size'],
#         num_layers=options['num_layers'],
#         num_keys=options['num_classes']
#     )

#     # --- 2) Load the data (! the attacker wouldnt have the access to data but would access it through training code and simply called this whole lsb function) ---
#     train_source = Path(options['data_dir']) / "hdfs_train_xu"   
#     original_sequences = load_training_sequences(str(train_source))

#     # --- 3) Load model weights (.pth) ---
#     ckpt_path = Path(cfg["checkpoint"]["path"])
#     assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
#     model_weights = torch.load(ckpt_path, map_location='cpu')
#     # keep your original guard: .pth can be raw state_dict or {"state_dict": ...}
#     if isinstance(model_weights, dict) and "state_dict" in model_weights:
#         model_weights = model_weights["state_dict"]

#     w = event_bit_width(options['num_classes'])  # from YAML
    

#     # --- 4) Encode ---
#     start_encoding = time.time()
#     new_weights = {}

#     if approach == "naive":
#         # --- Encode ---
#         binary_sequences = convert_sequences_to_binary(original_sequences)  
#         data_bits = ''.join(binary_sequences)
#         total_bits_to_encode = len(data_bits)

#         # Padding
#         remainder = len(data_bits) % n_bits
#         if remainder != 0:
#             data_bits += '0' * (n_bits - remainder)

#         remaining_bits = data_bits
#         new_weights = {}
#         for name, param in model_weights.items():
#             if remaining_bits:
#                 new_param, remaining_bits, _ = insert_data_into_tensor_all_bits(
#                     param, remaining_bits, replace_last_n=n_bits
#                 )
#             else:
#                 new_param = param
#             new_weights[name] = new_param

#         encoding_time = time.time() - start_encoding

#         # Save + evaluate
#         out_dir, attacked_ckpt_path = save_attacked_model(new_weights, cfg, options, n_bits, approach)
#         P, R, F1, FP, FN, prediction_time = evaluate_model(Model, options, attacked_ckpt_path)

#         # Decode & verify
#         start_decoding = time.time()
#         bitstream_exfiltrated = decode_all_parameters(new_weights, n_bits, total_bits_to_encode)
#         decoded_sequences = bits_to_sequences(bitstream_exfiltrated, len(original_sequences), len(original_sequences[-1]))
#         results = verify_decoding_all_metrics(decoded_sequences, original_sequences,
#                                             save_path_prefix=str(out_dir / "eval_results"),
#                                             known_events=options['num_classes'])
#         similarity, distance, compared_len = write_bitstreams(out_dir, data_bits, bitstream_exfiltrated)
#         decoding_time = time.time() - start_decoding

# # with naive insertion and extraction, the approach works but only when all data is inserted, if we try with n_bits = 2,
# # again ecc_decode doesnt work because encoded bitstring must be a multiple of 8 bits 
#     elif approach == "smart":

#         nsym = cfg["attack"]["params"]["nsym_payload"]
        
#         # --- Encode ---
#         # 1. We will encode only unique sequences to utilize model's capacity
#         unique_sequences, counts = deduplicate_and_sort_sequences(original_sequences)
#         data_bits, count_width, w, stats = serialize_sequences(unique_sequences, counts, options['num_classes'])
            
      
#         # 2) compute capacity 
#         target_params = select_least_sensitive_params(model_weights)
#         p_avail = params_available(target_params)
#         model_capacity = p_avail * n_bits
#         print(f"Total parameters: {p_avail}, capacity = {model_capacity} bits")

#         # 3) pack ECC stream to capacity
#         ecc_stream, used_plain, overhead_bits, overhead_ratio, header_spans = pack_ecc_stream(data_bits, nsym=nsym, capacity_bits=model_capacity)

        
#         # 4) write sequentially
#         param_list = collect_slots(model_weights, include_names=None)
        
#         header_n_bits = cfg["attack"]["params"]["header_top_k"]                 # how many top LSBits to use for headers (e.g., 1 -> use bit index n_bits-1)
 
        
#         spread_mapping, used_bits = bits_to_spread_mapping(ecc_stream, param_list, n_bits)        

#         new_weights = insert_data_smart(target_params, spread_mapping, n_bits, slots=param_list, header_spans=header_spans, header_top_k=header_n_bits)
#         encoding_time = time.time() - start_encoding

#         # Save + evaluate
#         out_dir, attacked_ckpt_path = save_attacked_model(new_weights, cfg, options, n_bits, approach)
#         #P, R, F1, FP, FN, prediction_time = evaluate_model(Model, options, attacked_ckpt_path)
#         start_decoding = time.time()

#         # 5) extract sequentially
#         exf_stream = extract_bits_sequential(new_weights, param_list, n_bits, header_spans=header_spans, header_top_k=header_n_bits)
#         hdr = exf_stream[:64]
#         print("hdr bits:", hdr)
#         print("orig_len raw bits:", hdr[:32])
#         print("orig_len int:", int(hdr[:32], 2))
#         print("enc_len raw bits:", hdr[32:])
#         print("enc_len int:", int(hdr[32:], 2))


#         # 6) ECC-unpack + decode (robust to prefix)
#         recovered_plain, decoding_metrics = unpack_ecc_stream(exf_stream, nsym=nsym)

#         # 7) deserialize
#         unique_sequences_extracted, counts_extracted, _ = deserialize_sequences(recovered_plain, count_width=0, w=w, num_classes=options["num_classes"])
#         decoding_time = time.time() - start_decoding

#         results = verify_decoding_smart(unique_sequences_extracted, counts_extracted, unique_sequences, counts,
#                                              save_path_prefix=str(out_dir / "eval_results"),
#                                              known_events=options['num_classes'])
#         print("decoding verified")

  
#         similarity, distance, compared_len = write_bitstreams(out_dir, data_bits, recovered_plain)



#     else:
#         raise ValueError(f"Unknown approach: {approach}")

    

#     # --- 9) Persist a report next to the attacked checkpoint ---
#     report = {
#         "attack": {
#             "type": "lsb",
#             "n_bits": n_bits,
#             "approach": approach,
# #            "capacity_bits": int(capacity),
#  #           "encoded_params_count": int(param_counter),
#  #           "bits_to_encode": int(total_bits_to_encode),
#  #           "payload_embedded": payload_embedded,
#   #          "payload_extracted": len(decoded_payload_concat),
#    #         "extraction_ratio": payload_embedded/len(decoded_payload_concat),

#         },
#         "metrics": {
#             # dataset that fits within model capacity (depends on n_bits)
#             "capacity": {
#                 "num_params": p_avail,
#                 "model_capacity": model_capacity,
#                 "total_dataset_size": len(data_bits),
#                 "coverage": min(1, (model_capacity/ len(data_bits))),
#                 "ecc_overhead_bits": overhead_bits,
#                 "ecc_overhead_ratio": overhead_ratio,
#             },

#             # entire dataset stats
#             "serialization": {
#                 "bits_events": stats.get("bits_events"),
#                 "bits_counts": stats.get("bits_counts"),
#                 "bits_delimiters": stats.get("bits_delimiters"),
#                 "total_bits_ser": stats.get("total_bits"),
#                 "n_sequences": stats.get("n_sequences"),
#                 "n_events": stats.get("n_events_total"),

#        #         "payload_bits_recovered": len(decoded_payload_concat),
#         #        "payload_recovery_rate": len(decoded_payload_concat) / len(data_bits) if len(data_bits) else 0.0
#             },

#             # "utility": {
#             #     "precision": float(P),
#             #     "recall": float(R),
#             #     "f1": float(F1),
#             #     "false_positives": int(FP),
#             #     "false_negatives": int(FN),
#             # },
#             "timing": {
#                 "encoding_time_s": float(encoding_time),
#   #              "prediction_time_s": float(prediction_time),
#                 "decoding_time_s": float(decoding_time),
#             },
#             "similarity": {
#                 "hamming_similarity": float(similarity),
#                 "hamming_distance": int(distance),
#                 "bits_compared": int(compared_len),
#                 # unpack your multi-metric verify() dict:
#                 "sequence_membership": results.get("sequence_membership"),
#                 "event_membership": results.get("event_membership"),
#                 "count_matches": results.get("count_matches"),
#                 "count_distance": results.get("count_distance"),
#                 "count_coverage": results.get("count_coverage"),
#                 "subsequence_coverage": results.get("subsequence_coverage"),

#             },
#             "decoding": {
#                 "total_chunks": decoding_metrics.get("chunks_total"),
#                 "chunks_decoded_ok": decoding_metrics.get("chunks_decoded_ok"),
#                 "decode_success_rate": decoding_metrics.get("decode_success_rate"),
#                 "useful_bits_intended": decoding_metrics.get("useful_bits_intended"),
#                 "useful_bits_decoded": decoding_metrics.get("useful_bits_decoded"),
#                 "dataset_coverage_decoded": decoding_metrics.get("dataset_coverage_decoded"),

#                 "raw_bits_used": decoding_metrics.get("raw_bits_used"),
#                 "enc_bits_total": decoding_metrics.get("enc_bits_total"),
#                 "header_bits_total": decoding_metrics.get("header_bits_total"),

#                 "overhead_bits_total": decoding_metrics.get("protocol_overhead_bits_total"),
#                 "overhead_ratio": decoding_metrics.get("overhead_ratio"),

#                 "per_chunk": decoding_metrics.get("per_chunk"),
#                 "stopped_on_incomplete_tail": decoding_metrics.get("stopped_on_incomplete_tail"),
#             },
#         },
#         "paths": {
#             "attacked_ckpt": str(attacked_ckpt_path),
#             "report_dir": str(out_dir),
#         },
        
#     }

#     with open(out_dir / "report.json", "w") as f:
#         json.dump(report, f, indent=2)

#     # --- 10) (Optional) Log to W&B if active, same keys as your original ---
#     try:
#         import wandb
#         wandb.init()
#         if wandb.run is not None:
#             wandb.log({
#               # --- attack info ---
#                 "attack/n_bits": n_bits,
#                 "attack/approach": approach,
    
#                 # --- capacity ---
#                 "capacity/num_params": p_avail,
#                 "capacity/model_capacity": model_capacity,
#                 "capacity/total_dataset_size": len(data_bits),
#                 "capacity/coverage": min(1, (model_capacity/ len(data_bits))),
#                 "capacity/ecc_overhead_bits": overhead_bits,
#                 "capacity/ecc_overhead_ratio": overhead_ratio,

#                 # --- serialization ---
#                 "serialization/bits_events": stats.get("bits_events"),
#                 "serialization/bits_counts": stats.get("bits_counts"),
#                 "serialization/bits_delimiters": stats.get("bits_delimiters"),
#                 "serialization/total_bits_ser": stats.get("total_bits"),
#                 "serialization/n_sequences": stats.get("n_sequences"),
#                 "serialization/n_events": stats.get("n_events_total"),
          
#                 # --- utility ---
#                 # "utility/precision": float(P),
#                 # "utility/recall": float(R),
#                 # "utility/f1": float(F1),
#                 # "utility/false_positives": int(FP),
#                 # "utility/false_negatives": int(FN),

#                 # --- timing ---
#                 "timing/encoding_time_s": float(encoding_time),
#            #     "timing/prediction_time_s": float(prediction_time),
#                 "timing/decoding_time_s": float(decoding_time),

#                 # --- similarity ---
#                 "similarity/hamming_similarity": float(similarity),
#                 "similarity/hamming_distance": int(distance),
#                 "similarity/bits_compared": int(compared_len),
#                 "similarity/sequence_membership": results.get("sequence_membership"),
#                 "similarity/event_membership": results.get("event_membership"),
#                 "similarity/count_matches": results.get("count_matches"),
#                 "similarity/count_distance": results.get("count_distance"),
#                 "similarity/count_coverage": results.get("count_coverage"),
#                 "similarity/subsequence_coverage": results.get("subsequence_coverage"),

#                 # --- decoding ---
#                 "decoding/total_chunks": decoding_metrics.get("chunks_total"),
#                 "decoding/chunks_decoded_ok": decoding_metrics.get("chunks_decoded_ok"),
#                 "decoding/decode_success_rate": decoding_metrics.get("decode_success_rate"),
#                 "decoding/useful_bits_intended": decoding_metrics.get("useful_bits_intended"),
#                 "decoding/useful_bits_decoded": decoding_metrics.get("useful_bits_decoded"),
#                 "decoding/dataset_coverage_decoded": decoding_metrics.get("dataset_coverage_decoded"),

#                 "decoding/raw_bits_used": decoding_metrics.get("raw_bits_used"),
#                 "decoding/enc_bits_total": decoding_metrics.get("enc_bits_total"),
#                 "decoding/header_bits_total": decoding_metrics.get("header_bits_total"),

#                 "decoding/overhead_bits_total": decoding_metrics.get("protocol_overhead_bits_total"),
#                 "decoding/overhead_ratio": decoding_metrics.get("overhead_ratio"),

#             })

#     except Exception:
#         pass

#     return report