import numpy as np
import scipy as sp
import scipy.stats as st
from collections import defaultdict
from scipy.spatial.distance import euclidean
import pdb

class LSH:

    def __init__(self, bin_distance, n_estimators=10, hash_size=7):
        self.n_estimators = n_estimators
        self.hash_size = hash_size
        self.w = bin_distance
        self.bin_distance = bin_distance

    def hash(self, point):
        result = []
        point = np.append(point, 1)
        for h in self.hashes:
            result.append(tuple((np.floor(np.sum(h * point, axis=1)/self.bin_distance)).astype(int)))
        return tuple(result)
    
    def insert(self, point):
        for est, hsh in zip(self.estimators, self.hash(point)):
            est[hsh].append(point)

    def fit(self, X):
        self.dim = len(X[0])
        self.hashes = [] 
        # dicts in python are hashtables so we don't have to implement them
        self.estimators = [defaultdict(lambda:[]) for i in range(self.n_estimators)]
        bin_distance = self.bin_distance
        for j in range(self.n_estimators):
            temp = []
            self.hashes.append(temp)
            for i in range(self.hash_size):
                temp.append(np.append(st.norm(0, 1).rvs(self.dim) / np.sqrt(self.dim), 
                                           st.uniform(-bin_distance, bin_distance).rvs(1)))
        for x in X:
            self.insert(x)

    def kneighbours(self, point, k):
        result = []
        for est, hsh in zip(self.estimators, self.hash(point)):
            result += est[hsh]
        result.sort(key=lambda x: euclidean(x, point))
        prev = None
        cleaned = []
        for i in range(len(result)):
            if prev is None or (prev != result[i]).any():
                cleaned.append(result[i])
            prev = result[i]
        return cleaned[:k]

if __name__ == '__main__':
    import numpy as np
    from lsh import LSH
    from scipy.stats import norm, uniform
    data = uniform(loc=0, scale=100).rvs(500 * 2).reshape((500, 2))
    index = LSH(10)
    index.fit(data)
    print(index.kneighbours(data[0], k=2))