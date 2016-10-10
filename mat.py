from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor
import struct

def ReadFile(filename):
    data = []
    lines = [line.rstrip('\n') for line in open(filename)]

    for line in lines:
        data.append(int(line))

    return data

def ReadBinary(filename, num):
    fin = open(filename, "rb")
    count = 0
    data = []
    while count < num:
        temp = struct.unpack('d', fin.read(8))
        data.append(temp[0])
        count = count + 1
    return data

def Restructure(array):
    n = int(sqrt(len(array)))
    mat = np.zeros((n,n), dtype=np.float64)
    count = 0
    row = 0
    col = 0
    for i in range(0, len(array)):
        mat[row][col] = array[i]
        col = col + 1
        if col == 3000:
            col = 0
            row = row + 1
    return mat

# Defining zeros as True will return the number of zeros instead
def NNZ(mat):
    count = np.count_nonzero(mat)
    return count

# start_i is the row index to start at
# start_j is the column index to start at
# m is the number of resulting columns
# n is the number of resulting rows
def SubMatrix(mat, n, m, start_i, start_j):
    new_mat = np.zeros((n,m))
    for i in range(0, n):
        for j in range(0, m):
            new_mat[i][j] = mat[start_i + i][start_j + j]
    return new_mat

def Rank(s):
    count = np.count_nonzero(s)
    return count

def FrobDiff(A, Ap):
    return np.linalg.norm(A - Ap)/np.linalg.norm(A)

def SVDTest(A, tol):
    A_cpy = A.copy()
    U, s, V = np.linalg.svd(A_cpy, full_matrices = False)
    print("\t Rank Before: " + str(len(s)))
    low_rank, rank = LowRankMat(U, s, V, tol)
    print("\t Rank After: " + str(rank))
    print("SVD Test: " + str(FrobDiff(A_cpy, low_rank)))

def ZeroDiagonals(mat):
    M = mat.copy()
    D = np.zeros((M.shape[0], M.shape[1]), dtype=np.float64)
    blk_sz = 375
    i_rng = []
    j_rng = []
    for par in range(0, int(M.shape[0]/blk_sz)):
        i_rng = []
        for n in range(0, blk_sz):
            i_rng.append(n + blk_sz * par)
        j_rng = i_rng
        for i in i_rng:
            for j in j_rng:
                D[i][j] = M[i][j]
                M[i][j] = 0
    return M, D

def LowRankMat(U, s, V, tol):
    global ind_2
    norm = np.linalg.norm(s, axis=0)
    U_mat = U.copy()
    V_mat = V.copy()
    norm_2 = norm
    s_rank = s.copy()
    index_back = -1
    while norm_2 > tol * norm:
        s_rank[index_back] = 0
        norm_2 = np.linalg.norm(s_rank, axis=0)
        index_back = index_back - 1
    S = np.diag(s_rank)
    low_mat = np.dot(U_mat, np.dot(S, V_mat))
    return low_mat, NNZ(s_rank)

# m is number of rows in low_rank
# n is number of columns in low_rank
# start_i is the row-index the submatrix starts at in mat
# start_j is the column-index the submatrix starts at in mat
def UpdateMat(mat, low_rank, start_i, start_j):
    temp_mat = mat.copy()
    for i in range(0, low_rank.shape[0]): # 0...m
        for j in range(0, low_rank.shape[1]): # 0...n
            temp_mat[i + start_i][j + start_j] = low_rank[i][j]
    return temp_mat

def RankSVD(mat, bins, tol):
    bins_3 = []
    ranks = []
    rank_diff = []
    nnz_mat = []
    nk2 = [] # Holds the reduced size of the sub matrix, 2nk
    for num in bins:
        bins_3.append(3 * num)
    start_i = 0
    start_j = 0
    cols = bins_3[:]
    rows = bins_3[:]
    for row in rows:
        for col in cols:
            sub_mat = SubMatrix(mat, row, col, start_i, start_j)
            n = sub_mat.shape[0]
            m = sub_mat.shape[1]
            U, s, V = np.linalg.svd(sub_mat, full_matrices = False)
            low_rank, rank = LowRankMat(U, s, V, tol)
            #print("\t" + str(FrobDiff(sub_mat, low_rank)))
            old_rank = NNZ(s)
            if rank < floor(old_rank / 2):
                nk2.append(2 * n * rank)
                mat = UpdateMat(mat, low_rank, start_i, start_j)
            else:
                nk2.append(n * m)
            rank_diff.append(old_rank - rank)
            ranks.append(rank)
            start_j = start_j + col
        start_i = start_i + row
        start_j = 0
    #print(rank_diff)
    return mat, nk2
