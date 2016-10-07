import numpy as np
from lowrank_mat import *

class HMat:
    def __init__(self, n, m):
        self.n = n # Rows
        self.m = m # Columns
        self.subMat = []

        def add_lowrank(self, low_rank):
            self.subMat.append(low_rank)

        def MatVec(self, x):
            result = np.zeros(x.shape[0], dtype=float64)
            for low_rank in subMat:
                for i in xrange(0, low_rank.n):
                    x_index = low_rank.start_i + i
                    for j in xrange(0, low_rank.d):
                        for k in xrange(0, low_rank.k):
                            result[x_index] = result[x_index] + low_rank.Y[i, k] * low_rank.Z[k, j] * x[x_index]
            return result
