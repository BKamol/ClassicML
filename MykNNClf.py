import numpy as np
import pandas as pd

class MyKNNClf:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight


    def __str__(self):
        return f"MyKNNClf class: k={self.k}"
    
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
        self.X_train = X.values
        self.y_train = y.values
        self.train_size = X.shape

    def predict(self, X):
        y_pred = [self._predict(x) for x in X.values]
        return np.array(y_pred)
    
    def predict_proba(self, X):
        y_pred = [self._predict(x, True) for x in X.values]
        return np.array(y_pred)

    def _predict(self, x, proba=False):
        dists = [self._distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(dists)[:self.k]
        k_nearest = [self.y_train[i] for i in k_idx]
        k_nearest_dists = [dists[i] for i in k_idx]

        Q0, Q1 = self._calcualte_weights(k_nearest, k_nearest_dists)

        if not proba:
            label = 1 if Q1 >= Q0 else 0
        else:
            label = Q1
        return label
    
    def _calculate_weights(self, k_nearest, k_nearest_dists):
        if self.weights == 'uniform':
            return (k_nearest.count(0)/self.k, k_nearest.count(1)/self.k)
        else:
            Ri0, Ri1, Ri = 0, 0, 0
            Di0, Di1, Di = 0, 0, 0
            for index, label in enumerate(k_nearest):
                dist = k_nearest_dists[index]
                if (label):
                    Ri1 += 1/(index+1)
                    Di1 += 1/dist
                else:
                    Ri0 += 1/(index+1)
                    Di0 += 1/dist
                Ri += 1/(index+1)
                Di += 1/dist
        if self.weight == 'rank':
            Q0, Q1 = Ri0/Ri, Ri1/Ri
        elif self.weight == 'distance':
            Q0, Q1 = Di0/Di, Di1/Di
        return (Q0, Q1)
    
        


