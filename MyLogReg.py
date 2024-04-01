import numpy as np
import random

class MyLogReg:
    
    def __init__(self, n_iter=10, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate if not isinstance(learning_rate, float) else lambda x: learning_rate
        self.weights = None
        self.metric = metric
        self.best_score = None

        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.sgd_sample = sgd_sample
        self.random_state = random_state

        self.eps = 1e-15
        self.sigmoid = lambda x: 1/(1+np.exp(-x))

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
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

            y_pred = Xi@self.weights
            y_pred =  self.sigmoid(y_pred)

            LogLoss = -1/n * np.sum(yi*np.log(y_pred+self.eps) + (1-yi)*np.log(1-y_pred+self.eps)) + self._reg()

            #printing log
            metr = self._calculate_metric(yi, y_pred) + self._reg()
            if (verbose and i == 0):
                print(f"start | loss: {LogLoss} | {self.metric} {metr}")
            elif (verbose and i % verbose == 0):
                print(f"{i} | loss: {LogLoss} | {self.metric} {metr}")

            grad = 1/n*(y_pred-yi)@Xi + self._grad_reg()
            self.weights -= self.learning_rate(i)*grad

        self.best_score = self._calculate_metric(y, self.sigmoid(X@self.weights))

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
        data = pd.DataFrame({'pred': y_pred, 'true': y}).sort_values(by='pred', ascending=False)
        y_sorted = np.array(data['true'])
        y_pred_sorted = np.array(data['pred'])
        result = 0
        class_1_index = np.where(y_sorted == 1)[0]
        for i in class_1_index:
            result += np.sum((y_sorted[i:] == 0) & (y_pred_sorted[i:] != y_pred_sorted[i]))
            result += np.sum((y_sorted[i:] == 0) & (y_pred_sorted[i:] == y_pred_sorted[i])) / 2

        return result / ( np.sum(y == 1) * np.sum(y == 0) )

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

    def get_best_score(self):
        return self.best_score