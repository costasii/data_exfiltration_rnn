from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Global config
# =========================

SPLITS = ["split0", "split1", "split2"]

BASE_DIRS = {
    "HDFS": Path("/home/mirta/dev/data_exfiltration/results/federated/hdfs/attack_bb_hdfs_final"),
    "BGL": Path("/home/mirta/dev/data_exfiltration/models/federated/attacked/bgl"),
}

SAVE_DIR = Path("./figures/attack_performance")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Experiment registry
# =========================

EXPERIMENTS = {
    "payload_encoding_strategy": {
        "title": "Payload encoding strategy",
        "variable": "round_schedule",
        "settings": {
            "same_data": {
                "label": "Same data every round",
                "round_schedule": "fixed_once",
                "cumulative": False,
                "folders": {
                    "HDFS": lambda split, c: f"{split}_c{c}_fixed_once_start0",
                    "BGL":  lambda split, c: f"attack_bb_bgl_{split}_le30_c{c}_fixed_once_start0",
                },
            },
            "new_data": {
                "label": "New data every round",
                "round_schedule": "new_each_round",
                "cumulative": True,
                "folders": {
                    "HDFS": lambda split, c: f"{split}_c{c}_new_each_round_start0",
                    "BGL":  lambda split, c: f"attack_bb_bgl_{split}_le30_c{c}_new_each_round_start0",
                },
            },
        },
    },
    "temporal_dynamics": {
        "title": "Temporal dynamics",
        "variable": "attack_start_round",
        "settings": {
            0: {
                "label": "Attack start round = 0",
                "attack_start_round": 0,
                "cumulative": False,
                "folders": {
                    "HDFS": lambda split, c: f"{split}_c{c}_fixed_once_start0",
                    "BGL":  lambda split, c: f"attack_bb_bgl_{split}_le30_c{c}_fixed_once_start0",
                },
            },
            5: {
                "label": "Attack start round = 5",
                "attack_start_round": 5,
                "cumulative": False,
                "folders": {
                    "HDFS": lambda split, c: f"{split}_c{c}_fixed_once_start5",
                    "BGL":  lambda split, c: f"attack_bb_bgl_{split}_le30_c{c}_fixed_once_start5",
                },
            },
        },
    },

    "num_clients": {
    "title": "Number of clients",
    "variable": "num_clients",
    "settings": {
        2: {
            "label": "2 clients",
            "num_clients": 2,
            "cumulative": False,
            "folders": {
                "HDFS": lambda split, c: f"{split}_c2_fixed_once_start0",
                "BGL":  lambda split, c: f"attack_bb_bgl_{split}_le30_c2_fixed_once_start0",
            },
        },
        10: {
            "label": "10 clients",
            "num_clients": 10,
            "cumulative": False,
            "folders": {
                "HDFS": lambda split, c: f"{split}_c10_fixed_once_start0",
                "BGL":  lambda split, c: f"attack_bb_bgl_{split}_le30_c10_fixed_once_start0_rep200",
            },
        },
        50: {
            "label": "50 clients",
            "num_clients": 50,
            "cumulative": False,
            "folders": {
                "HDFS": lambda split, c: f"attack_bb_hdfs_{split}_le10_c50_fixed_once_start0_rep200",
                "BGL":  lambda split, c: f"attack_bb_bgl_{split}_le30_c50_fixed_once_start0_rep200",
            },
        },
    },
},

    
}

EXPERIMENTS["local_epochs"] = {
    "settings": {
        10: {
            "label": "10 local epochs",
            "num_clients": 2,
            "cumulative": False,
            "folders": {
                "HDFS": lambda split, num_clients: f"{split}_c2_fixed_once_start0",
            },
        },
        100: {
            "label": "100 local epochs",
            "num_clients": 2,
            "cumulative": False,
            "folders": {
                "HDFS": lambda split, num_clients: f"{split}_c2_fixed_once_start0local_ep100",
            },
        },
    }
}

EXPERIMENTS["num_repetitions"] = {
    "settings": {
        200: {
            "label": "200 repetitions",
            "num_clients": 2,
            "cumulative": False,
            "folders": {
                "BGL": lambda split, num_clients: f"attack_bb_bgl_{split}_le30_c2_fixed_once_start0",
                "HDFS": lambda split, num_clients: f"{split}_c2_fixed_once_start0",
            },
        },
        500: {
            "label": "500 repetitions",
            "num_clients": 2,
            "cumulative": False,
            "folders": {
                "BGL": lambda split, num_clients: f"attack_bb_bgl_{split}_le30_c2_fixed_once_start0_rep500",
                "HDFS": lambda split, num_clients: f"attack_bb_hdfs_{split}_le10_c2_fixed_once_start0_rep500",
            },
        },
    }
}