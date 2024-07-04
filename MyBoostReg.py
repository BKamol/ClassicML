import numpy as np
import pandas as pd

class Node:
    def __init__(self):
        self.feature = None
        self.value_split = None
        self.value_leaf = None
        self.side = None
        self.left = None
        self.right = None
        self.idx = None

class MyTreeReg:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 1
        self.bins = bins
        self.__sum_tree_values = 0
        self.split_values = {}
        self.criterion = criterion
        self.fi = {}

    def fit(self, X, y):
        self.tree = None
        self.fi = { col: 0 for col in X.columns }
        
        def create_tree(root, X_root, y_root, side='root', depth=0):
            if root is None:
                root = Node()
            col_name, split_value, ig = self.get_best_split(X_root, y_root)

            mean_value = y_root.mean()

            if depth >= self.max_depth or \
              len(y_root) < self.min_samples_split or \
              (self.leafs_cnt > 1 and self.leafs_cnt >= self.max_leafs):
                root.side = side
                root.value_leaf = mean_value
                root.idx = list(y_root.index)
                self.__sum_tree_values += root.value_leaf
                return root

            self.fi[col_name] += len(y_root) / len(y) * ig

            X_left = X_root.loc[X_root[col_name] <= split_value]
            y_left = y_root.loc[X_root[col_name] <= split_value]

            X_right = X_root.loc[X_root[col_name] > split_value]
            y_right = y_root.loc[X_root[col_name] > split_value]

            if len(X_left) == 0 or len(X_right) == 0:
                root.side = side
                root.value_leaf = mean_value
                root.idx = list(y_root.index)
                self.__sum_tree_values += root.value_leaf
                return root

            root.feature = col_name
            root.value_split = split_value
            self.leafs_cnt += 1

            root.left = create_tree(root.left, X_left, y_left, 'left', depth + 1)
            root.right = create_tree(root.right, X_right, y_right, 'right', depth + 1)

            return root

        self.tree = create_tree(self.tree, X, y)

    def predict(self, X):
        y_pred = []
        for _, row in X.iterrows():
            node = self.tree
            while node.feature is not None:
                if row[node.feature] <= node.value_split:
                    node = node.left
                else:
                    node = node.right
            y_pred.append(node.value_leaf)
        return np.array(y_pred)
    
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node.feature is not None:
            print(f"{' ' * depth}{node.feature} > {node.value_split}")
            if node.left is not None:
                self.print_tree(node.left, depth + 1)
            if node.right is not None:
                self.print_tree(node.right, depth + 1)
        else:
            print(f"{' ' * depth}{node.side} = {node.value_leaf}")

    def get_best_split(self, X, y):
        mse_0 = self.mse(y)

        col_name = None
        split_value = None
        gain = -float('inf')

        for col in X.columns:
            if not (col in self.split_values.keys()):
                x_unique_values = np.unique(X[col])
                if self.bins is None or len(x_unique_values) - 1 < self.bins:
                    self.split_values[col] = np.array([(x_unique_values[i - 1] + \
                    x_unique_values[i]) / 2 for i in range(1, len(x_unique_values))])
                else:
                    _, self.split_values[col] = np.histogram(X[col], bins=self.bins)

            for split_value_i in self.split_values[col]:
                mask = X[col] <= split_value_i
                left_split, right_split = y[mask], y[~mask]

                mse_left = self.mse(left_split)
                mse_right = self.mse(right_split)

                weight_left = len(left_split) / len(y)
                weight_right = len(right_split) / len(y)

                mse_i = weight_left * mse_left + weight_right * mse_right

                gain_i = mse_0 - mse_i
                if gain < gain_i:
                    col_name = col
                    split_value = split_value_i
                    gain = gain_i

        return col_name, split_value, gain
            
    def mse(self, t):
        t_mean = np.mean(t)
        return np.sum((t - t_mean) ** 2) / (len(t)+1e-15)
    
    def __node_rule(self, p, split=pd.Series()):
        if self.criterion == 'entropy':
            return -np.sum(p * np.log2(p)) if not split.empty else 0
        elif self.criterion == 'gini':
            return 1 - np.sum(p ** 2)

    def __str__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}"
    
    def sum_leafs(self):
        return self.__sum_tree_values
    
    def replace_leafs(self, X, y, loss="MSE"):
        queue = []
        queue.append(self.tree)
        while (queue):
            node = queue.pop(0)
            if (node and node.value_leaf):
                idx = node.idx
                y_pred = self.predict(X.loc[idx, :])
                loss_value = y[idx] - y_pred
                node.value_leaf = loss_value.mean() if loss == 'MSE' else loss_value.median()
            queue.append(node.left)
            queue.append(node.right)


class MyBoostReg:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=5, min_samples_split=2, max_leafs=20, bins=16, loss='MSE', metric=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.loss = loss
        self.metric = metric

        self.pred_0 = None
        self.trees = []
        self.best_score = None

    def __repr__(self):
        return f'MyBoostReg class: n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}'

    def calc_score(self, y, y_pred):
        if self.metric == 'MSE':
            return np.sum((y - y_pred)**2) / len(y)
        elif self.metric == 'MAE':
            return np.sum(np.abs((y - y_pred))) / len(y)
        elif self.metric == 'RMSE':
            return np.sqrt(np.sum((y - y_pred)**2) / len(y)) 

    def fit(self, X, y):
        self.pred_0 = y.mean() if self.loss == 'MSE' else y.median()
        Fm = self.pred_0
        for _ in range(self.n_estimators):
            rm = 2*(Fm - y)
            tree = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins)
            tree.fit(X, rm)
            tree.replace_leafs(X, y, self.loss)
            self.trees.append(tree)
            y_pred = self.learning_rate * tree.predict(X)
            Fm += y_pred

        y_pred = self.predict(X)
        self.best_score = self.metric(y, y_pred)
    
    def predict(self, X):
        y_pred = 0
        for tree in self.trees:
            y_pred += tree.predict(X)
        y_pred = self.learning_rate * y_pred + self.pred_0
        return y_pred





