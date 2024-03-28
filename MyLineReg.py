import numpy as np
import random

class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate if not isinstance(learning_rate, float) else lambda x: learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None

        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        
        self.sgd_sample = sgd_sample
        self.random_state = random_state


    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)

        n, n_feats = len(y), X.shape[1]+1
        I = np.ones(n)
        X = np.c_[I, X]

        if self.sgd_sample is not None:
            n = self.sgd_sample if isinstance(self.sgd_sample, int) else round(self.sgd_sample * n)
        
        self.weights = np.ones(n_feats)
        #Gradient Descent
        for i in range(1, self.n_iter+1):
            #Subset of X and y in case of stochastic gradient descent
            if self.sgd_sample is not None:
                sample_rows_idx = random.sample(range(X.shape[0]), n)
                Xi = X[sample_rows_idx, :]
                yi = y.iloc[sample_rows_idx]
            else:
                Xi, yi = X, y
            
            y_pred = Xi@self.weights #
            ##Loss calculations
            loss = 1/n * np.sum((y_pred - yi)**2) + self._reg()
            metr = self._calculate_metric(yi, y_pred, self.metric) + self._reg()
            if (verbose and i == 0):
                print(f"start | loss: {loss} | {self.metric} {metr}")
            if (verbose and i % verbose == 0):
                print(f"{i} | loss: {loss} {self.metric} {metr}")
                
            #Step towards antigradient
            grad = 2/n * (y_pred - yi)@Xi + self._grad_reg()
            self.weights -= self.learning_rate(i)*grad


        self.best_score = self._calculate_metric(y, X@self.weights, self.metric)

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

    @staticmethod
    def _calculate_metric(y, y_pred, metric):
        formulas = {'mae': lambda y, y_pred: 1/len(y)*np.sum(np.abs(y_pred - y)),
                    'mse': lambda y, y_pred: 1/len(y) * np.sum((y_pred - y)**2),
                    'rmse': lambda y, y_pred: np.sqrt(1/len(y) * np.sum((y_pred - y)**2)),
                    'mape': lambda y, y_pred: 100/len(y)*np.sum(np.abs((y - y_pred)/y)),
                    'r2': lambda y, y_pred: 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
                    }
        return formulas[metric](y, y_pred) if metric is not None else formulas['mse'](y, y_pred)
    
    def _reg(self):
        regs = {'l1': lambda l1, l2, weights: l1*np.sum(np.abs(weights)),
                'l2': lambda l1, l2, weights: l2*np.sum(weights**2),
                'elasticnet': lambda l1, l2, weights: l1 * np.sum(np.abs(weights)) + l2 * np.sum(weights**2)
                }
        return regs[self.reg](self.l1_coef, self.l2_coef, self.weights) if self.reg is not None else 0

    def _grad_reg(self):
        grad_regs = {'l1': lambda l1, l2, weights: l1*np.sign(weights),
                     'l2': lambda l1, l2, weights: l2*2*(weights),
                     'elasticnet': lambda l1, l2, weights: l1*np.sign(weights) + l2*2*(weights)
                     }
        return grad_regs[self.reg](self.l1_coef, self.l2_coef, self.weights) if self.reg is not None else 0





