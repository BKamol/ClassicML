import numpy as np
import pandas as pd


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        

class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

        self.leafs_cnt = 0
        self.root = None
        self.bins = bins
        self.splitters = None
        self.criterion = criterion


    def __repr__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def _get_splitters(self, x):
        values = np.array(sorted(x.unique()))
        splitters = np.array([(values[i] + values[i+1])/2 for i in range(len(values)-1)])
        return splitters
    
    def _get_splitters_wbins(self, X, y):
        self.splitters = {}
        if self.bins:
            native_splitters = {col: self._get_splitters(X[col]) for col in X.columns}
            for col, splitters in native_splitters.items():
                if len(splitters) <= self.bins - 1:
                    self.splitters[col] = splitters
                else:
                    hist = np.histogram(X[col], bins=self.bins)[1][1:-1]
                    self.splitters[col] = hist

    def _entropy(self, col_targ):
        p0 = (col_targ.iloc[:, 1] == 0).sum() / (col_targ.shape[0] + 1e-15)
        p1 = col_targ.iloc[:, 1].sum() / (col_targ.shape[0] + 1e-15)
        S = -p0*np.log2(p0+1e-15) - p1*np.log2(p1+1e-15)
        return S

    def _gini(self, col_targ):
        p0 = (col_targ.iloc[:, 1] == 0).sum() / (col_targ.shape[0] + 1e-15)
        p1 = col_targ.iloc[:, 1].sum() / (col_targ.shape[0] + 1e-15)
        G = 1 - p0**2 - p1**2
        return G

    def _get_ig(self, x, y, split):
        col_targ = pd.concat([x, y], axis=1)

        left_sub = col_targ.loc[col_targ[x.name] <= split, :]
        right_sub = col_targ.loc[col_targ[x.name] > split, :]

        if self.criterion == 'gini':
            Gp = self._gini(col_targ)
            Gl, Gr = self._gini(left_sub), self._gini(right_sub)
            IG = Gp - left_sub.shape[0]/(col_targ.shape[0] + 1e-15)*Gl - right_sub.shape[0]/(col_targ.shape[0] + 1e-15)*Gr
        else:
            S0 = self._entropy(col_targ)
            S1, S2 = self._entropy(left_sub), self._entropy(right_sub)
            IG = S0 - left_sub.shape[0]/(col_targ.shape[0] + 1e-15)*S1 - right_sub.shape[0]/(col_targ.shape[0] + 1e-15)*S2
        return IG
        

    def _get_best_split(self, X, y):
        cols = X.columns
        if self.bins is None:
            splitters = {col: self._get_splitters(X[col]) for col in cols}
        else:
            splitters = self.splitters
        best_col = None
        best_split = None
        best_ig = 0
        for col, splits in splitters.items():
            x = X[col]
            igs = np.array([self._get_ig(x, y, split) for split in splits])
            max_idx = igs.argmax()
            max_ig, max_split = igs[max_idx], splits[max_idx]
            if max_ig > best_ig:
                best_col = col
                best_split = max_split
                best_ig = max_ig
        
        return best_col, best_split, best_ig

    def is_leaf(self, data, depth):
        return (all(data.iloc[:, -1] == 1)) or\
               (all(data.iloc[:, -1] == 0)) or\
               (depth >= self.max_depth-1) or\
               (data.shape[0] < self.min_samples_split) or\
               (self.leafs_cnt >= self.max_leafs-1)

    
    def _fit(self, X, y, depth=0):
        best_col, best_split, best_ig = self._get_best_split(X, y)
        root = Node((best_col, best_split))
        #print(best_col, best_split, best_ig)
        col_targ = pd.concat([X, y], axis=1)

        if (best_col is None):
            value = col_targ.iloc[:, -1].sum() / col_targ.shape[0]
            return Node(('leaf', value))

        left_sub = col_targ.loc[col_targ[best_col] <= best_split, :]
        right_sub = col_targ.loc[col_targ[best_col] > best_split, :]

        if self.is_leaf(left_sub, depth):
            value = left_sub.iloc[:, -1].sum() / (left_sub.shape[0] + 1e-15)
            root.left = Node(('left', value))
            self.leafs_cnt += 1
        else:
            X, y = left_sub.drop(left_sub.columns[-1], axis=1), left_sub.iloc[:, -1]
            root.left = self._fit(X, y, depth+1)

        if self.is_leaf(right_sub, depth):
            value = right_sub.iloc[:, -1].sum() / (right_sub.shape[0] + 1e-15)
            root.right = Node(('right', value))
            self.leafs_cnt += 1
        else:
            X, y = right_sub.drop(right_sub.columns[-1], axis=1), right_sub.iloc[:, -1]
            root.right = self._fit(X, y, depth+1)
        return root

    def fit(self, X, y):
        self._get_splitters_wbins(X, y)
        self.root = self._fit(X, y)

    
    def _predict_proba(self, x, root):
        if (root.right is None and root.left is None):
            return root.value[1]
        
        feat, split = root.value
        if x[feat] <= split:
            return self._predict_proba(x, root.left)
        else:
            return self._predict_proba(x, root.right)

    def predict_proba(self, X):
        y_pred_logits = np.array([self._predict_proba(X.iloc[i, :], self.root) for i in range(X.shape[0])])
        return y_pred_logits

    def predict(self, X):
        y_pred = (self.predict_proba(X) > 0.5).astype(int)
        return y_pred


    def _print_tree(self, root, intend):
        if (root is None):
            return None
        
        feat, split = root.value
        if (root.left is None and root.right is None):
            print('  '*intend, end='')
            print(f"{feat} = {split}")
        else:
            print('  '*intend, end='')
            print(f"{feat} > {split}")

        self._print_tree(root.left, intend+1)
        self._print_tree(root.right, intend+1)

    def print_tree(self):
        self._print_tree(self.root, 0)

    def _sum_leafs(self, root):
        if (root is None):
            return 0
        if (root.left is None and root.right is None):
            return root.value[1]
        return self._sum_leafs(root.left) + self._sum_leafs(root.right)
    
    def sum_leafs(self):
        return self._sum_leafs(self.root)





