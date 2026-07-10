import flwr as fl
import sys

from flwr.common import parameters_to_ndarrays
import json
import os
from datetime import datetime, timezone
import torch
# Add project root directory to path for models import
server_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(server_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.mlp import GraphAwareMLP


# Custom strategy subclassing FedAvg for server-side early stopping
class EarlyStoppingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, patience=2, min_delta=0.0002, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = 0.0
        self.patience_counter = 0

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters_aggregated is not None:
            ndarrays = parameters_to_ndarrays(parameters_aggregated)
            self.last_weights = [arr.tolist() for arr in ndarrays]
            self.last_client_count = len(results)
            
            # Serialize the global model weights into PyTorch format (model.pt)
            try:
                model = GraphAwareMLP(input_dim=32)
                model.set_parameters(ndarrays)
                checkpoint_path = os.path.join(project_root, "fl_implementation", "model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"💾 PyTorch model checkpoint saved successfully to {checkpoint_path}")
            except Exception as e:
                print(f"⚠️ Error saving model checkpoint: {e}")
                
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Write the aggregated weights and metrics to json
        if hasattr(self, 'last_weights'):
            metrics_dict = metrics_aggregated if metrics_aggregated is not None else {}
            export_data = {
                "round": server_round,
                "weights": self.last_weights,
                "metrics": {
                    "accuracy": metrics_dict.get("accuracy", 0.0),
                    "precision": metrics_dict.get("precision", 0.0),
                    "recall": metrics_dict.get("recall", 0.0),
                    "f1": metrics_dict.get("f1", 0.0)
                },
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "client_count": getattr(self, 'last_client_count', 0)
            }
            
            server_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(server_dir)
            schemas_dir = os.path.join(project_root, "schemas")
            os.makedirs(schemas_dir, exist_ok=True)
            
            export_path = os.path.join(schemas_dir, f"global_model_round_{server_round}.json")
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)
            print(f"💾 Exported global model for round {server_round} to {export_path}")
        
        if metrics_aggregated is None:
            return loss_aggregated, metrics_aggregated
            
        current_f1 = metrics_aggregated.get("f1", 0.0)
        
        # Check if F1 has improved significantly
        if current_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = current_f1
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        print(f"📊 Early Stopping Status: Patience {self.patience_counter}/{self.patience} | Best F1: {self.best_f1:.4f} | Current F1: {current_f1:.4f}")
        
        if self.patience_counter >= self.patience:
            print("\n🛑 EARLY STOPPING TRIGGERED: Global F1-score has saturated (no significant improvement).")
            print(f"🏆 Final Aggregated F1 Score: {self.best_f1:.4f}")
            print("Stopping federated learning loop successfully...")
            sys.exit(0)
            
        return loss_aggregated, metrics_aggregated

if __name__ == "__main__":
    print("🚀 Starting FL Server...")
    # 🔹 Custom metric aggregation function
    def weighted_average(metrics):
        total_examples = sum(num_examples for num_examples, _ in metrics)

        aggregated = {}

        for num_examples, m in metrics:
            for k, v in m.items():
                aggregated[k] = aggregated.get(k, 0) + v * num_examples

        for k in aggregated:
            aggregated[k] /= total_examples

        # 🔥 CLEAN PRINT
        print("\n📊 ROUND SUMMARY")
        print(f"Accuracy : {aggregated['accuracy']:.4f}")
        print(f"Precision: {aggregated['precision']:.4f}")
        print(f"Recall   : {aggregated['recall']:.4f}")
        print(f"F1 Score : {aggregated['f1']:.4f}")
        print("-" * 30)

        return aggregated


    # 🔹 Strategy with metric aggregation (configured to wait for all 4 bank clients)
    strategy = EarlyStoppingFedAvg(
        patience=2,
        min_delta=0.0002,
        evaluate_metrics_aggregation_fn=weighted_average,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
    )


    # 🔹 Start server with max rounds set to 10 (early stopping will shut it down once saturated)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
    # fl.server.start_server(
    #     server_address="0.0.0.0:8080",
    #     config=fl.server.ServerConfig(num_rounds=3),
    # )