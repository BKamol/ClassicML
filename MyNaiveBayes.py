import numpy as np


class MyNaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = y.unique()
        n_classes = len(self._classes)
        self._means = np.zeros((n_classes, n_features))
        self._vars = np.zeros((n_classes, n_features))
        self._priors = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self._classes):
            X_c = X[y==c]
            self._means[idx] = X_c.mean(axis=0)
            self._vars[idx] = X_c.var(axis=0)
            self._priors[idx] = len(X_c) / len(y)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            posterior = np.log(self._priors[idx])
            posterior += np.sum(np.log(self._pdf(idx, x)))
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, idx, x):
        return 1/np.sqrt(2*np.pi*self._vars[idx]**2)*np.exp(-(x - self._means[idx])**2 / 2*(self._vars[idx]**2))
