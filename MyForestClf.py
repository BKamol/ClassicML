import numpy as np 
import pandas as pd
import random, copy
from MyTreeClf import MyTreeClf

class MyForestClf:
    def __init__(self, 
                 n_estimators=10, 
                 max_features=0.5, 
                 max_samples=0.5, 
                 random_state=42,
                 max_depth=5,
                 min_samples_split=2,
                 max_leafs=20,
                 bins=16,
                 criterion='entropy'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion

        self.leafs_cnt = 0
        self.trees = []
        self.fi = {}

    def __repr__(self):
        return f'MyForestClf class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, criterion={self.criterion}, random_state={self.random_state}'
    
    def __sum_fi(self):
        for tree in self.trees:
            for key, value in tree.fi.items():
                self.fi[key] += value

    def fit(self, X, y):
        random.seed(self.random_state)
        cols_smpl_cnt = round(self.max_features * X.shape[1])
        rows_smpl_cnt = round(self.max_samples * X.shape[0])
        for _ in range(self.n_estimators):
            cols_idx = random.sample(list(X.columns), cols_smpl_cnt)
            rows_idx = random.sample(range(X.shape[0]), rows_smpl_cnt)
            X_subset = X.loc[rows_idx, cols_idx]
            y_subset = y[rows_idx]
            tree = MyTreeClf(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, self.criterion)
            tree.fit(X_subset, y_subset, len(X))
            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt

        self.fi = {col: 0 for col in X.columns}
        self.__sum_fi()

            

    def _predict(self, x, type):
        y_preds = np.array([tree.predict_proba(x) for tree in self.trees])
        if type == 'mean':
            return y_preds.mean()
        else:
            y_preds = y_preds > 0.5
            ones = y_preds.sum()
            zeros = len(y_preds) - ones
            return 1 if ones > zeros else 0

    def predict_proba(self, X):
        y_pred = np.array([self._predict(x.to_frame().T, 'mean') for _, x in X.iterrows()])
        return y_pred

    def predict(self, X, type='mean'):
        y_pred = np.array([self._predict(x.to_frame().T, type) for _, x in X.iterrows()])
        if type == 'mean':
            y_pred = y_pred > 0.5
        return y_pred


    