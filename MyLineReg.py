import numpy as np

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def fit(self, X, y, verbose=False):
        n, n_feats = len(y), X.shape[1]+1
        I = np.ones(n)
        X = np.c_[I, X]

        self.weights = np.ones(n_feats)
        for i in range(self.n_iter):
            y_pred = X@self.weights

            mse = 1/n * np.sum((y_pred - y)**2)
            metric = self._calculate_metric(y, y_pred)
            if (verbose and i == 0):
                print(f"start | loss: {mse} | {self.metric} {metric}")
            if (verbose and i % verbose == 0):
                print(f"{i} | loss: {mse} {self.metric} {self._calculate_metric(y, y_pred)}")
            
            grad = 2/n * (y_pred - y)@X
            self.weights -= self.learning_rate*grad

        self.best_score = self._calculate_metric(y, X@self.weights)

    def predict(self, X):
        n, n_feats = X.shape[0], X.shape[1]+1
        I = np.ones(n)
        X = np.c_[I, X]

        y_pred = X@self.weights
        return y_pred

    def get_coef(self):
        return np.array(self.weights[1:])
    
    def get_best_score(self):
        return self.best_score

    def _calculate_metric(self, y, y_pred):
        formulas = {'mae': lambda y, y_pred: 1/len(y)*np.sum(np.abs(y_pred - y)),
                    'mse': lambda y, y_pred: 1/len(y) * np.sum((y_pred - y)**2),
                    'rmse': lambda y, y_pred: np.sqrt(1/len(y) * np.sum((y_pred - y)**2)),
                    'mape': lambda y, y_pred: 100/len(y)*np.sum(np.abs((y - y_pred)/y)),
                    'r2': lambda y, y_pred: 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)}
        return formulas[self.metric](y, y_pred)


