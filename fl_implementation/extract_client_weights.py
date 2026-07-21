
import sys
import os
import json

import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.mlp import GraphAwareMLP


def extract_output_layer_params(checkpoint_path, input_dim=32):
    model = GraphAwareMLP(input_dim=input_dim)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    state = model.state_dict()

    weight = state["output_layer.0.weight"].numpy().flatten()
    bias = state["output_layer.0.bias"].numpy().flatten()

    params = weight.tolist() + bias.tolist()
    if len(params) != 33:
        raise AssertionError(
            f"Expected 33 output-layer params, got {len(params)} from {checkpoint_path}."
        )
    return params


def count_rows(csv_path):
    return len(pd.read_csv(csv_path))


def main():
    args = sys.argv[1:]
    if len(args) != 8:
        print(__doc__)
        sys.exit(1)

    pairs = [(args[i], args[i + 1]) for i in range(0, 8, 2)]

    client_weights = []
    client_counts = []
    for checkpoint_path, data_path in pairs:
        print(f"Extracting from {checkpoint_path} ...")
        client_weights.append(extract_output_layer_params(checkpoint_path))
        n_rows = count_rows(data_path)
        client_counts.append(n_rows)
        print(f"  {data_path}: {n_rows} rows")

    with open("clients.json", "w") as f:
        json.dump({"clientWeights": client_weights, "clientCounts": client_counts}, f, indent=2)

    print(f"\nWrote clients.json: {len(pairs)} banks, real counts {client_counts}")


if __name__ == "__main__":
    main()