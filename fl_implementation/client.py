from sklearn.linear_model import LogisticRegression

import flwr as fl
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


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
        

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = LogisticRegression(
            penalty='l2',
            solver='saga',        # IMPORTANT for large data
            max_iter=1,           # match FL rounds
            warm_start=True,      # reuse weights
            n_jobs=-1,
            random_state=42
        )
        # self.model.fit(self.X[:10], self.y[:10])  # init
        # Ensure both classes present in init
        init_idx = []
        classes = set()

        for i, label in enumerate(self.y):
            if label not in classes:
                init_idx.append(i)
                classes.add(label)
            if len(classes) == 2:
                break

        self.model.fit(self.X[init_idx], self.y[init_idx])

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        # n_layers = len(self.model.coefs_)
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        

        probs = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (probs > 0.3).astype(int)

        accuracy = self.model.score(self.X_test, self.y_test)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        loss = 1 - accuracy

        return loss, len(self.X_test), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="server:8080",
        client=AMLClient(),
    )