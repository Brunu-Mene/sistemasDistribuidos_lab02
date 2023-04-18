import flwr as fl

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    acc = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    results = {"accuracy": sum(acc) / sum(examples)}
    return results


if __name__ == '__main__':
    strategy = fl.server.strategy.FedAvg(min_available_clients=5, evaluate_metrics_aggregation_fn=weighted_average)

    server_address = "[::]:8081"
    fl.server.start_server(server_address=server_address, config = fl.server.ServerConfig(num_rounds=10), strategy = strategy)