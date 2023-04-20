import flwr as fl
import utils
# rounds2 = 'accuracy': [(1, 0.9449000000953675), (2, 0.9703000068664551)]}
# rounds5 = accuracy': [(1, 0.9514999866485596), (2, 0.9700000047683716), (3, 0.9765999913215637), (4, 0.98089998960495), (5, 0.9823000073432923)]}
# rounds10 = 'accuracy': [(1, 0.9375999927520752), (2, 0.9681999921798706), (3, 0.9743000030517578), (4, 0.9784999966621399), (5, 0.980400002002716), (6, 0.9805999994277954), (7, 0.9831000089645385), (8, 0.9842999935150146), (9, 0.9853000044822693), (10, 0.9855999946594238)]}
# rounds20 = accuracy': [(1, 0.9437999963760376), (2, 0.9653999924659729), (3, 0.972700011730194), (4, 0.9777000069618225), (5, 0.9810000061988831), (6, 0.9823000073432923), (7, 0.9834999918937684), (8, 0.9845999956130982), (9, 0.9850000023841858), (10, 0.9859000086784363), (11, 0.9857999801635742), (12, 0.9870000004768371), (13, 0.986899995803833), (14, 0.9871999979019165), (15, 0.9866999864578248), (16, 0.9871999979019165), (17, 0.9873000025749207), (18, 0.9883000016212463), (19, 0.9876999974250793), (20, 0.9882999897003174)]}

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