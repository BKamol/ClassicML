import numpy as np
import pandas as pd
import random, copy
from MyTreeReg import MyTreeReg

class MyForestReg:
    def __init__(self, 
                 n_estimators=10, 
                 max_features=0.5, 
                 max_samples=0.5, 
                 random_state=42,
                 max_depth=5,
                 min_samples_split=2,
                 max_leafs=20,
                 bins=16):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        self.leafs_cnt = 0
        self.trees = []
        self.fi = {}

    def __repr__(self):
        return f"MyForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, random_state={self.random_state}"

    def _sum_fi(self):
        for tree in self.trees:
            for feat, imp in tree.fi.items():
                self.fi[feat] += imp

    def fit(self, X, y):
        random.seed(self.random_state)
        cols_smpl_cnt = round(self.max_features * X.shape[1])
        rows_smpl_cnt = round(self.max_samples * X.shape[0])
        for _ in range(self.n_estimators):
            cols_idx = random.sample(list(X.columns), cols_smpl_cnt)
            rows_idx = random.sample(range(X.shape[0]), rows_smpl_cnt)
            X_subset = X.loc[rows_idx, cols_idx]
            y_subset = y[rows_idx]
            tree = MyTreeReg(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_leafs=self.max_leafs, bins=self.bins)
            tree.fit(X_subset, y_subset, len(X))
            self.trees.append(tree)
            self.leafs_cnt += tree.leafs_cnt
        
        self.fi = {col: 0 for col in X.columns}
        self._sum_fi()
        

    def _predict(self, x):
        y_preds = np.array([tree.predict(x) for tree in self.trees])
        return y_preds.mean()

    def predict(self, X):
        return np.array([self._predict(x.to_frame().T) for _, x in X.iterrows()])
        


