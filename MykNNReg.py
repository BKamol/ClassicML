import numpy as np
import pandas as pd

class MyKNNReg:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.train_size = None
        self.metric = metric

    def __str__(self):
        return f"MyKNNReg class: k={self.k}"
    
    @staticmethod
    def _euclidian(x, y):
        return np.sqrt(np.sum((x - y)**2))
    
    @staticmethod
    def _chebyshev(x, y):
        return max(abs(x-y))
    
    @staticmethod
    def _manhattan(x, y):
        return sum(abs(x-y))
    
    @staticmethod
    def _cosine(x, y):
        return 1 - (x@y.T)/( np.sqrt(x@x.T * y@y.T) )
    
    def _distance(self, x, y):
        funcs = {
            'euclidean': self._euclidian,
            'chebyshev': self._chebyshev,
            'manhattan': self._manhattan,
            'cosine': self._cosine
        }
        return funcs[self.metric](x, y)
    
    def fit(self, X, y):
        self.X_train = X.values.copy()
        self.y_train = y.values.copy()
        self.train_size = X.shape

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x, proba=False):
        dists = [self._distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(dists)[:self.k]
        k_nearest = np.array([self.y_train[i] for i in k_idx])
        return k_nearest.mean()


