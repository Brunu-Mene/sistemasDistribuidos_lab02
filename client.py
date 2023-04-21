import flwr as fl
import utils

if __name__ == '__main__':
    input_shape = (28, 28, 1)
    num_classes = 10
    cid = 4
    n = 5

    x_train, x_test, y_train, y_test = utils.loadData(cid,n)
    model = utils.define_model(input_shape,num_classes)

    class FlowerClient(fl.client.NumPyClient):
        def __init__(self, model, x_train, y_train, x_test, y_test) -> None:
            self.model = model
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test

        def get_parameters(self, config):
            return self.model.get_weights()

        def fit(self, parameters, config):
            self.model.set_weights(parameters)
            self.model.fit(self.x_train, self.y_train, epochs=1, verbose=2)
            return self.model.get_weights(), len(self.x_train), {}

        def evaluate(self, parameters, config):
            self.model.set_weights(parameters)
            loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=2)
            return loss, len(self.x_test), {"accuracy": acc}
    
    client = FlowerClient(
        model,
        x_train,
        y_train,
        x_test,
        y_test
    )

    server_address = "[::]:8081"
    fl.client.start_numpy_client(server_address=server_address, client=client)