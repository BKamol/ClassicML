import numpy as np
import pandas as pd

class MyLogReg:
    
    def __init__(self, n_iter=10, learning_rate=0.1, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.best_score = None
        self.eps = 1e-15

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def fit(self, X, y, verbose=False):
        n, n_feats = len(y), X.shape[1]+1
        I = np.ones(n)
        X = np.c_[I, X]

        self.weights = np.ones(n_feats)

        #Gradient Descent
        for i in range(self.n_iter):
            y_pred = X@self.weights
            y_pred = 1/(1+np.exp(-y_pred)) #sigmoid

            LogLoss = -1/n * np.sum(y*np.log(y_pred+self.eps) + (1-y)*np.log(1-y_pred+self.eps))

            #printing log
            metr = self._calculate_metric(y, y_pred)
            if (verbose and i == 0):
                print(f"start | loss: {LogLoss} | {self.metric} {metr}")
            elif (verbose and i % verbose == 0):
                print(f"{i} | loss: {LogLoss} | {self.metric} {metr}")

            grad = 1/n*(y_pred-y)@X
            self.weights -= self.learning_rate*grad

        self.best_score = self._calculate_metric(y, X@self.weights)

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X):
        n, n_feats = X.shape[0], X.shape[1]+1
        I = np.ones(n)
        X = np.c_[I, X]
        y_pred = X@self.weights
        y_pred = 1/(1+np.exp(-y_pred)) #sigmoid
        return y_pred

    def predict(self, X):
        y_pred = np.array([x > 0.5 for x in self.predict_proba(X)])
        return y_pred

        
    def _calculate_metric(self, y, y_pred):
        if (self.metric != 'roc_auc'):
            y_pred = np.array([1 if x > 0.5 else 0 for x in y_pred])
        else:
            return self._roc_auc(y, y_pred)
        TP = np.sum((y_pred == 1) & (y == 1))
        FP = np.sum((y_pred == 1) & (y == 0))
        FN = np.sum((y_pred == 0) & (y == 1))
        TN = np.sum((y_pred == 0) & (y == 0))
        precision = TP/(TP+FP+self.eps)
        recall = TP/(TP+FN+self.eps)
        formulas = {'accuracy': (TP + TN)/(TP+TN+FP+FN),
                    'precision': precision,
                    'recall': recall,
                    'f1': 2*precision*recall/(precision+recall+self.eps),
                    }
        return formulas[self.metric] if self.metric is not None else 0

    @staticmethod
    def _roc_auc(y, y_pred):
        y_pred = pd.Series(y_pred).round(10)
        df = pd.concat([y_pred, y], axis=1)
        df = df.sort_values(by=0, ascending=False)
        P = (df[1] == 1).sum()
        N = (df[1] == 0).sum()
        total = 0
        for _, row in df.iterrows():
            if row[1] == 0:
                score_higher = df[df[0] > row[0]]
                total += score_higher[score_higher[1] == 1][1].count()
                score_equal = df[df[0] == row[0]]
                total += score_equal[score_equal[1] == 1][1].count() / 2
        return total / (P * N)

    def get_best_score(self):
        return self.best_score