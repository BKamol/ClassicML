import numpy as np

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def fit(self, X, y, verobe=False):
        I = np.zeros(X.shape[0])
        I[0] = 1
        X = np.c_[I, X]
        self.weights = np.ones(X.shape[0])
        for _ in range(self.n_iter):
            y_pred = X@self.weights
            mse = 1/len(y) * np.sum((y_pred - y)**2)
            


