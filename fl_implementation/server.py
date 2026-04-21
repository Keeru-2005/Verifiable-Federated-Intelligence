import flwr as fl

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


    # 🔹 Strategy with metric aggregation
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
    )


    # 🔹 Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
    # fl.server.start_server(
    #     server_address="0.0.0.0:8080",
    #     config=fl.server.ServerConfig(num_rounds=3),
    # )