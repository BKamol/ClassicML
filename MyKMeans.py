import numpy as np
import pandas as pd

class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=10, n_init=3, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = []

    def __repr__(self):
        return f"MyKMeans class: n_clusters={self.n_clusters}, max_iter={self.max_iter}, n_init={self.n_init}, random_state={self.random_state}"
    
    def __distance(self, x, y):
        return np.sqrt(np.sum((x - y)**2))

    def __generate_centroids(self, X):
        np.random.seed(seed=self.random_state)
        centroids = []
        for _ in range(self.n_clusters):
            centroid = np.array([np.random.uniform(low=X[feat].min(), high=X[feat].max()) for feat in X.columns])
            centroids.append(centroid)
        return np.array(centroids)
    
    def __distances(self, x, centroids):
        return np.array([self.__distance(x, centroid) for centroid in centroids])

    def __get_new_centroids(self, clusters, X):
        clusters_dict = {cluster: [] for cluster in range(max(clusters)+1)}
        for i in range(len(clusters)):
            clusters_dict[clusters[i]].append(i)
        
        new_centroids = []
        for cluster, rows in clusters_dict.items():
            centroid = np.array([X.loc[rows, feat].mean() for feat in X.columns])
            new_centroids.append(centroid)
        return np.array(new_centroids)


    def _fit(self, X):
        centroids = self.__generate_centroids(X)
        for _ in range(self.max_iter):
            clusters = [self.__distances(x, centroids).argmin() for x in X.values]
            new_centroids = self.__get_new_centroids(clusters, X)
            
            if (new_centroids == centroids).all():
                break
        if np.nan not in new_centroids:
            self.centroids.append(new_centroids)
    
    def __wcss(self, centroids, X):
        return np.sum([self.__distances(x, centroids).min()**2 for x in X.values])
        
    def __best_centroids(self, X):
        wcsss = np.array([self.__wcss(centroids, X) for centroids in self.centroids])
        print(wcsss)
        return self.centroids[wcsss.argmin()], wcsss.min()


    def fit(self, X):
        for _ in range(self.n_init):
            self._fit(X)
        self.cluster_centers_, self.inertia_ = self.__best_centroids(X)

    def predict(self, X):
        clusters = [self.__distances(x, self.cluster_centers_).argmin() for x in X.values]
        return np.array(clusters)

