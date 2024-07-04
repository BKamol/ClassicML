import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=100):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                condition = y_[idx] * (x @ self.w - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * 2 * self.lambda_param * self.w
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - x @ y_[idx])
                    self.b -= self.learning_rate * y_[idx]


    def predict(self, X):
        y_pred = X @ self.w - self.b
        return np.sign(y_pred)
