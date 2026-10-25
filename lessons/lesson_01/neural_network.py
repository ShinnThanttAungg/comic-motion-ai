import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=64, hidden_size=16, learning_rate=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, 1))
        self.b2 = np.zeros((1, 1))
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def train_step(self, X, y):
        m = len(X)
        predictions = self.forward(X)
        eps = 1e-12
        loss = -np.mean(y * np.log(predictions + eps) + (1 - y) * np.log(1 - predictions + eps))
        dZ2 = (predictions - y) / m
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = (dZ2 @ self.W2.T) * (self.Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        return float(loss)

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
