import flwr as fl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import os

DATA_FILE = os.getenv("DATA_FILE")

def load_data():
    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=['is_laundering'])
    y = df['is_laundering']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


class AMLClient(fl.client.NumPyClient):
    def __init__(self):
        self.X, self.y = load_data()
        self.model = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=1)
        self.model.fit(self.X[:10], self.y[:10])  # init

    def get_parameters(self, config):
        return self.model.coefs_ + self.model.intercepts_

    def set_parameters(self, parameters):
        n_layers = len(self.model.coefs_)
        self.model.coefs_ = parameters[:n_layers]
        self.model.intercepts_ = parameters[n_layers:]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X, self.y)
        return self.get_parameters(config), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = 1 - self.model.score(self.X, self.y)
        return loss, len(self.X), {"accuracy": 1 - loss}


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="server:8080",
        client=AMLClient(),
    )