# src/pipelines/federated/dataset_builder.py
from src.data.log import log_dataset
from src.data.sliding_window import sliding_window_from_sequences

def build_client_dataset(sequences, opts):
    logs, labels = sliding_window_from_sequences(
        sequences=sequences,
        window_size=opts["window_size"],
        subtract_1=opts.get("subtract_1", True),
    )

    dataset = log_dataset(
        logs=logs,
        labels=labels,
        seq=opts["sequentials"],
        quan=opts["quantitatives"],
        sem=opts["semantics"],
    )
    return dataset