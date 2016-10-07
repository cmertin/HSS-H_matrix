from __future__ import division, print_function
from lowrank_mat import *
from mat import FrobDiff
import numpy as np

n = 100
m = 100
k = 40

A = np.random.rand(n,m)
x = np.random.rand(m)
U, s, V = np.linalg.svd(A, full_matrices = False)

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

print(Yk.shape, Vk.shape)

Ak = np.dot(Yk, Vk)

Ak_x = np.dot(Ak, x)

Ax = np.dot(A, x)

relErr = FrobDiff(A, Ak)
Ax_k_norm = (np.linalg.norm(Ax - Ak_x))/np.linalg.norm(x)

print(s[k-1])

compress = LowRankMat(0, 0, n, k, m, Yk, Vk)

Cx = MatVec(compress, x)

Cx_norm = (np.linalg.norm(Ax - Cx))/np.linalg.norm(x)

print(compress.n)

print(relErr)
print(Ax_k_norm)
print(Cx_norm)
