import flwr as fl


if __name__ == '__main__':
    strategy = fl.server.strategy.FedAvg(min_available_clients=2)
    fl.server.ServerConfig

    server_address = "[::]:8081"
    fl.server.start_server(server_address=server_address, config = fl.server.ServerConfig(num_rounds=15), strategy = strategy)