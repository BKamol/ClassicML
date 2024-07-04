import numpy as np

class MyPCA:
    def __init__(self, n_components=3):
        self.n_components = n_components

    def __repr__(self):
        return f"MyPCA class: n_components={self.n_components}"

    def fit_transform(self, X):
        X_meaned = X - X.mean()
        cov_mat = X_meaned.cov()
        W_pca = np.linalg.eigh(cov_mat)[1][:, -self.n_components:]
        X_reduced = X_meaned @ W_pca
        return X_reduced

        