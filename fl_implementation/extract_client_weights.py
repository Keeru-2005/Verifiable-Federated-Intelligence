"""Prepare proof inputs from the exact Flower results for one completed round."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("round", type=int, help="Completed Flower round to prepare for proving")
    parser.add_argument("--schemas-dir", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schemas_dir = args.schemas_dir or os.environ.get("VFI_SCHEMAS_DIR") or os.path.join(project_root, "schemas")
    manifest_path = os.environ.get(
        "VFI_MANIFEST_PATH",
        os.path.join(project_root, "data", "preprocess_manifest.json"),
    )
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError("Current preprocess manifest not found; rerun preprocessing first.")
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("pipeline_version") != "causal-v3" or manifest.get("status") != "complete":
        raise ValueError("Proof generation requires causal-v3 preprocessing artifacts.")
    source_path = os.path.join(schemas_dir, "round_updates", f"round_{args.round}.json")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Flower round update record not found: {source_path}")

    with open(source_path, encoding="utf-8") as source_file:
        source = json.load(source_file)
    if int(source.get("round", -1)) != args.round:
        raise ValueError("Requested round does not match the saved Flower update record.")
    if source.get("run_id") != manifest.get("run_id"):
        raise ValueError("Flower update record belongs to a different preprocessing run.")

    clients = source.get("clients", [])
    if len(clients) != 4:
        raise ValueError(f"The compiled circuit requires exactly 4 clients; found {len(clients)}.")
    client_weights = []
    client_counts = []
    for client in clients:
        count = int(client.get("num_examples", 0))
        weights = client.get("output_layer_parameters", [])
        if count <= 0 or len(weights) != 33:
            raise ValueError("Each client record must contain positive Flower num_examples and 33 output-layer parameters.")
        client_counts.append(count)
        client_weights.append([float(weight) for weight in weights])

    global_weights = source.get("global_output_layer_parameters", [])
    if len(global_weights) != 33:
        raise ValueError("Saved global model record must contain its 33 output-layer parameters.")
    total = sum(client_counts)
    for index, actual in enumerate(global_weights):
        expected = sum(client_weights[c][index] * client_counts[c] for c in range(4)) / total
        if abs(expected - float(actual)) > 2e-5:
            raise ValueError(
                f"Saved Flower updates do not reproduce the round aggregate at output parameter {index}: "
                f"expected {expected}, found {actual}."
            )

    proof_source = {
        "round": args.round,
        "run_id": manifest.get("run_id"),
        "clientWeights": client_weights,
        "clientCounts": client_counts,
        "globalOutputLayerParameters": [float(x) for x in global_weights],
    }
    out_path = os.path.join(schemas_dir, "round_updates", f"circuit_round_{args.round}.json")
    with open(out_path, "w", encoding="utf-8") as out_file:
        json.dump(proof_source, out_file, indent=2)
    print(f"Prepared exact round-{args.round} proof inputs: {out_path}")
    print(f"Flower sample counts: {client_counts} (total {total})")


if __name__ == "__main__":
    main()
