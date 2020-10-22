import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
class KMeans2:
    def __init__(self, k,max_iter=300):
        self.K = k
        self.max_iters = max_iter
    def fit(self,X):
        centroids = X[np.random.choice(np.arange(len(X)), self.K,replace = False), :]
        cluster = np.argmin(distance.cdist(X,centroids,'euclidean'),axis=1)
        for i in range(self.max_iters):
            centroids = np.vstack([X[cluster==i,:].mean(axis=0) for i in range(self.K)])
            tmp = np.argmin(distance.cdist(X,centroids,'euclidean'),axis=1)
            if np.array_equal(cluster,tmp):break
            cluster=tmp
        
        self.cluster_centers_ = centroids
        self.inertia_ =  np.sum(distance.cdist(X,centroids,'euclidean'))
