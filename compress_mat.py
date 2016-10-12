import numpy as np
from lowrank_mat import *

class HMat:
    def __init__(self, n, m):
        self.n = n # Rows
        self.m = m # Columns
        self.subMat = []

    def add_lowrank(self, mat, k, start_i = 0, start_j = 0):
        low_rank = CompressMatrixID_2(mat, k, start_i, start_j)
        self.subMat.append(low_rank)

    def MatVec(self, x):
        result = np.zeros(x.shape[0], dtype=np.float64)
        for low_rank in self.subMat:
            for i in xrange(0, low_rank.n):
                x_index = low_rank.start_i + i
                for j in xrange(0, low_rank.d):
                    for k in xrange(0, low_rank.k):
                        result[x_index] = result[x_index] + low_rank.Y[i, k] * low_rank.Z[k, j] * x[x_index]
        return result

    def GetSubMat(self):
        return self.subMat


def Output_Hmat(Hmat, filename):
    output = open(filename,'w')

    line = str(Hmat.n) + "," + str(Hmat.m) + "\n"
    output.write(line)
    output.write(str(len(Hmat.subMat)) + "\n")
    for sub_mat in Hmat.subMat:
        n = sub_mat.n
        k = sub_mat.k
        m = sub_mat.d
        start_i = sub_mat.start_i
        line = str(start_i) + "," + str(n) + "," + str(k) + "," + str(m) + "\n"
        output.write(line)
        for i in range(0, n):
            line = ""
            for j in range(0, k):
                if j != k-1:
                    line = line + str(sub_mat.Y[i][j]) + ","
                else:
                    line = line + str(sub_mat.Y[i][j]) + "\n"
            output.write(line)
        for i in range(0, k):
            line = ""
            for j in range(0, m):
                if j != m-1:
                    line = line + str(sub_mat.Z[i][j]) + ","
                else:
                    line = line + str(sub_mat.Z[i][j]) + "\n"
            output.write(line)
    output.close()

def Output_Vec(x, filename):
    output = open(filename, 'w')

    m = str(x.shape[0]) + "\n"
    output.write(m)
    for i in range(0, m):
        ele = str(x[i]) + "\n"
        output.write(ele)
    output.close()
