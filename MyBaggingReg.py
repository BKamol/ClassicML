import numpy as np
import pandas as pd
import random, copy

class MyBaggingReg:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0, random_state=42):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators = []

    def __repr__(self):
        return f"MyBaggingReg class: estimator={self.estimator}, n_estimators={self.n_estimators}, max_samples={self.max_samples}, random_state={self.random_state}"

    def fit(self, X, y):
        random.seed(self.random_state)
        rows_num_list = list(range(len(X)))
        rows_smpl_cnt = round(len(X) * self.max_samples)

        samples_idx = []
        for _ in range(self.n_estimators):
            sample_rows_idx = random.choices(rows_num_list, k=rows_smpl_cnt)
            samples_idx.append(sample_rows_idx)
        
        for i in range(self.n_estimators):
            X_sample = X.loc[samples_idx[i], :]
            y_sample = y[samples_idx[i]]
            model = copy.copy(self.estimator)
            model.fit(X_sample, y_sample)
            self.estimators.append(model)
            

    def _predict(self, x):
        return np.array([estimator.predict(x) for estimator in self.estimators]).mean()

    def predict(self, X):
        return np.array([self._predict(x.to_frame().T) for _, x in X.iterrows()])


