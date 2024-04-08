import numpy as np

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class MyTreeClf:
    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.root = None

    def __repr__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"
    
    @staticmethod
    def prob(y):
        return np.sum(y) / len(y)



    def _fit(self, X, y, depth):
        #Splitting data by best feature 
        best_feat, best_split, ig = self.get_best_split(X, y)
        X['y'] = y
        left = X.loc[X[best_feat] <= best_split, :]
        right = X.loc[X[best_feat] > best_split, :]
        left_X, left_y = left.drop('y', axis=1), left['y']
        right_X, right_y = right.drop('y', axis=1), right['y']
        root = Node((best_feat, best_split))    #saving best split in root


        if (depth < self.max_depth-1) and (len(left_y) >= self.min_samples_split) and (self.leafs_cnt < self.max_leafs-2) and (self.prob(left_y) not in (0,1)):
            root.left = self._fit(left_X, left_y, depth+1)   #recursively split left node
        else:
            root.left = Node(("leaf_left", self.prob(left_y)))
            self.leafs_cnt += 1

        if ((depth < self.max_depth-1) and len(right_y) >= self.min_samples_split) and (self.leafs_cnt < self.max_leafs-2) and (self.prob(right_y) not in (0,1)):
            root.right = self._fit(right_X, right_y, depth+1)   # recursivly split right node
        else:
            root.right = Node(("leaf_right", self.prob(right_y)))
            self.leafs_cnt += 1

        return root
        

    def fit(self, X, y):
        self.root = self._fit(X, y, 0)

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


    @staticmethod
    def entropy(x):
      p = np.bincount(x) / len(x)
      return -1 * np.sum(p * np.log2(p + 1e-15))

    def information_gain(self, x, y):
        N = len(x) + len(y)
        s0 = self.entropy(np.concatenate((x, y)))
        sx, sy = self.entropy(x), self.entropy(y)
        return s0 - len(x) / N * sx - len(y) / N * sy

    def get_best_split(self, X, y):
        cols = X.columns.tolist()
        X, y = np.asarray(X), np.asarray(y)
        col_name, split_value, ig = None, 0, 0
        for col_index in range(X.shape[1]):
            values = X[:, col_index]
            uniques = np.sort(np.unique(values))
            ents = []
            for index in range(len(uniques)):
                sep = np.mean(uniques[index : index + 2])
                current_ig = self.information_gain(y[values <= sep], y[values > sep])

                if current_ig > ig:
                    col_name = cols[col_index]
                    split_value = sep
                    ig = current_ig

        return col_name, split_value, ig

