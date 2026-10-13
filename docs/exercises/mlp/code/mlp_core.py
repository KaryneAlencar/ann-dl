
import numpy as np


class MLP:
    def __init__(self, layers, task="binary", rng=None):
        self.layers = layers
        self.task = task

        if rng is None:
            raise ValueError("É necessário fornecer o gerador rng.")

        self.W = [
            rng.normal(0, np.sqrt(2 / (a + b)), size=(a, b))
            for a, b in zip(layers[:-1], layers[1:])
        ]

        self.b = [
            np.zeros((1, b))
            for b in layers[1:]
        ]

    def forward(self, X):
        acts = [X]

        for W, b in zip(self.W[:-1], self.b[:-1]):
            acts.append(np.tanh(acts[-1] @ W + b))

        z = acts[-1] @ self.W[-1] + self.b[-1]

        if self.task == "binary":
            out = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        else:
            z = z - z.max(axis=1, keepdims=True)
            ex = np.exp(z)
            out = ex / ex.sum(axis=1, keepdims=True)

        acts.append(out)
        return acts

    def fit(self, X, y, lr=0.15, epochs=1500):
        if self.task == "binary":
            Y = y.reshape(-1, 1)
        else:
            Y = np.eye(self.layers[-1])[y]

        history = []
        n = len(X)

        for epoch in range(epochs):
            a = self.forward(X)
            out = np.clip(a[-1], 1e-12, 1 - 1e-12)

            if self.task == "binary":
                loss = -np.mean(
                    Y * np.log(out)
                    + (1 - Y) * np.log(1 - out)
                )
            else:
                loss = -np.mean(
                    np.sum(Y * np.log(out), axis=1)
                )

            history.append(float(loss))

            delta = (a[-1] - Y) / n

            gradsW = [None] * len(self.W)
            gradsb = [None] * len(self.b)

            for i in range(len(self.W) - 1, -1, -1):
                gradsW[i] = a[i].T @ delta
                gradsb[i] = delta.sum(axis=0, keepdims=True)

                if i > 0:
                    delta = (
                        delta @ self.W[i].T
                    ) * (1 - a[i] ** 2)

            for i in range(len(self.W)):
                self.W[i] -= lr * gradsW[i]
                self.b[i] -= lr * gradsb[i]

        return np.array(history)

    def predict(self, X):
        out = self.forward(X)[-1]

        if self.task == "binary":
            return (out[:, 0] >= 0.5).astype(int)

        return out.argmax(axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def count_params(self):
        return sum(
            w.size + b.size
            for w, b in zip(self.W, self.b)
        )