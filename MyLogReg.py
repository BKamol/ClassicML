import numpy as np

class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    
    def fit(self, X, y, verbose=False, eps=1e-15):
        n, n_feats = len(y), X.shape[1]+1
        I = np.ones(n)
        X = np.c_[I, X]

        self.weights = np.ones(n_feats)

        #Gradient Descent
        for i in range(self.n_iter):
            y_pred = X@self.weights
            y_pred = 1/(1+np.exp(-y_pred)) #sigmoid

            LogLoss = -1/n * np.sum(y*np.log(y_pred+eps) + (1-y)*np.log(1-y_pred+eps))

            #printing log
            if (verbose and i == 0):
                print(f"start | loss: {LogLoss}")
            elif (verbose and i % verbose == 0):
                print(f"{i} | loss: {LogLoss}")


            grad = 1/n*(y_pred-y)@X
            self.weights -= self.learning_rate*grad

    def get_coef(self):
        return self.weights[1:]