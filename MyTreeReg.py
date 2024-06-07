import numpy as np
import pandas as pd


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        

class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

        self.leafs_cnt = 0
        self.root = None

    def __repr__(self):
        return f"MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"


    def _get_splitters(self, x):
        values = np.array(sorted(x.unique()))
        splitters = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]
        return np.array(splitters)

    def _get_mse(self, col_targ):
        y = col_targ.iloc[:, -1]
        return np.sum((y - y.mean())**2) / (len(y) + 1e-15)

    def _get_ig(self, x, y, split):
        col_targ = pd.concat([x, y], axis=1)

        left_sub = col_targ.loc[col_targ[x.name] <= split, :]
        right_sub = col_targ.loc[col_targ[x.name] > split, :]

        Np, Nl, Nr = col_targ.shape[0], left_sub.shape[0], right_sub.shape[0]
        MSEp, MSEl, MSEr = self._get_mse(col_targ), self._get_mse(left_sub), self._get_mse(right_sub)
        IG = MSEp - (Nl/Np*MSEl + Nr/Np*MSEr)
        return IG

    def _get_best_split(self, X, y):
        splitters = {col: self._get_splitters(X[col]) for col in X.columns}

        best_col = None
        best_split = None
        best_ig = 0
        for col, splits in splitters.items():
            igs = np.array([self._get_ig(X[col], y, split) for split in splits])
            if not len(igs): continue
            max_ig_idx = igs.argmax()
            max_ig = igs[max_ig_idx]
            max_split = splits[max_ig_idx]
            if max_ig > best_ig:
                best_col = col
                best_split = max_split
                best_ig = max_ig
        return best_col, best_split, best_ig

    def is_leaf(self, data, depth):
        return (data.iloc[:, -1].unique().shape[0] <= 1) or\
               (depth >= self.max_depth-1) or\
               (data.shape[0] < self.min_samples_split-1) or\
               (self.leafs_cnt >= self.max_leafs-1)
    
    def _fit(self, X, y, depth=0):
        best_col, best_split, best_if = self._get_best_split(X, y)
        root = Node((best_col, best_split))

        col_targ = pd.concat([X, y], axis=1)

        left_sub = col_targ.loc[col_targ[best_col] <= best_split, :]
        right_sub = col_targ.loc[col_targ[best_col] > best_split, :]

        if self.is_leaf(left_sub, depth):
            value = left_sub.iloc[:, -1].mean()
            root.left = Node(('left', value))
            self.leafs_cnt += 1
        else:
            X, y = left_sub.drop(left_sub.columns[-1], axis=1), left_sub.iloc[:, -1] 
            root.left = self._fit(X, y, depth+1)

        if self.is_leaf(right_sub, depth):
            value = right_sub.iloc[:, -1].mean()
            root.right = Node(('right', value))
            self.leafs_cnt += 1
        else:
            X, y = right_sub.drop(right_sub.columns[-1], axis=1), right_sub.iloc[:, -1]
            root.right = self._fit(X, y, depth+1)

        return root
            
    def fit(self, X, y):
        self.root = self._fit(X, y)


    def _predict(self, x, root):
        if root.left is None and root.right is None:
            return root.value[1]
        if x[root.value[0]] <= root.value[1]:
            return self._predict(x, root.left)
        else:
            return self._predict(x, root.right)

    def predict(self, X):
        y_pred = [self._predict(X.iloc[i, :], self.root) for i in range(len(X))]
        return np.array(y_pred)

    
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

