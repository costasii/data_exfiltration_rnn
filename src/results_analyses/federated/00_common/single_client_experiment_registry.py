# single_client_experiment_registry.py

from pathlib import Path

SPLITS = ["split0", "split1", "split2"]

BASE_DIRS = {
    "HDFS": Path(
        "/home/mirta/dev/data_exfiltration/models/federated/attacked/hdfs"
    ),
    "BGL": Path(
        "/home/mirta/dev/data_exfiltration/models/federated/attacked/bgl"
    ),
}
SAVE_DIR = Path("./figures/single_client_attack")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_CLIENT_EXPERIMENTS = {
    "num_clients": {
        "title": "Number of federated clients",
        "variable": "num_clients",
        "settings": {
            5: {
                "label": "5 clients",
                "num_clients": 5,
                "attacked_client_id": 0,
                "cumulative": False,
                "folders": {
                    "HDFS": lambda split: (
                        f"attack_bb_hdfs_{split}_le10_c5_"
                        f"fixed_once_start0_rep200"
                    ),
                    "BGL": lambda split: (
                        f"attack_bb_bgl_{split}_le30_c5_"
                        f"fixed_once_start0_rep200"
                    ),
                },
            },
            10: {
                "label": "10 clients",
                "num_clients": 10,
                "attacked_client_id": 0,
                "cumulative": False,
                "folders": {
                    "HDFS": lambda split: (
                        f"attack_bb_hdfs_{split}_le10_c10_"
                        f"fixed_once_start0_rep200"
                    ),
                    "BGL": lambda split: (
                        f"attack_bb_bgl_{split}_le30_c10_"
                        f"fixed_once_start0_rep200"
                    ),
                },
            },
            50: {
                "label": "50 clients",
                "num_clients": 50,
                "attacked_client_id": 0,
                "cumulative": False,
                "folders": {
                    "HDFS": lambda split: (
                        f"attack_bb_hdfs_{split}_le10_c50_"
                        f"fixed_once_start0_rep200"
                    ),
                    "BGL": lambda split: (
                        f"attack_bb_bgl_{split}_le30_c50_"
                        f"fixed_once_start0_rep200"
                    ),
                },
            },
        },
    },
}