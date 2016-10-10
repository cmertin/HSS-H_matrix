import numpy as np

class LowRankMatrix:
    def __init__(self, start_i, start_j, n, k, d, Y, Z):
        self.start_i = start_i
        self.start_j = start_j
        self.n = n # Rows
        self.k = k # Rank
        self.d = d # Columns
        self.Y = Y # U_k * S_k
        self.Z = Z # V_k^T

# A is a compressed matrix, x is a full-sized vector
def MatVec(A, x):
    sol = np.zeros(A.n)
    for i in xrange(0, A.n):
        for j in xrange(0, A.d):
            for k in xrange(0, A.k):
                sol[i] = sol[i] + A.Y[i, k] * A.Z[k, j] * x[j]
    return sol

def CompressMatrix(A, k, start_i = 0, start_j = 0):
    U, s, V = np.linalg.svd(A, full_matrices = False)

    n = A.shape[0]
    m = A.shape[1]

    Uk = np.zeros((n,k), dtype=np.float64)
    Sk = np.zeros((k,k), dtype=np.float64)
    Vk = np.zeros((k,m), dtype=np.float64)

    for i in xrange(0, Uk.shape[0]):
        for j in xrange(0, Uk.shape[1]):
            Uk[i,j] = U[i,j]

    for i in xrange(0, Sk.shape[0]):
        Sk[i,i] = s[i]

    for i in xrange(0, Vk.shape[0]):
        for j in xrange(0, Vk.shape[1]):
            Vk[i,j] = V[i,j]

    Yk = np.dot(Uk, Sk)
    low_rank = LowRankMatrix(start_i, start_j, n, k, m, Yk, Vk)
    return low_rank
