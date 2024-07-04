import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.activation_fuction = lambda x: np.where(x > 0, 1, 0)


    def fit(self, X, y):
        n_features, n_samples = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                y_pred = self.activation_fuction(self.weights @ x + self.bias)
                update = self.learning_rate * (y_[idx] - y_pred)
                self.weights += update * x
                self.bias += update


    def predict(self, X):
        y_pred = self.activation_fuction(self.weights @ X + self.bias)
        return y_pred