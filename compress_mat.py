import numpy as np
from lowrank_mat import *
import os

class HMat:
    def __init__(self, n, m):
        self.n = n # Rows
        self.m = m # Columns
        self.subMat = []

    def add_lowrank(self, mat, k, start_i = 0, start_j = 0, minRank = False):
        #low_rank = CompressMatrixID_2(mat, k, start_i, start_j)
        if(minRank == False):
            low_rank = CompressMatrix(mat, k, start_i, start_j)
        else:
            n = mat.shape[0]
            m = mat.shape[1]
            k = m
            Z = np.identity(m)
            low_rank = LowRankMatrix(start_i, start_j, n, k, m, mat, Z)
        self.subMat.append(low_rank)

    def MatVec(self, x, hmat_file, vec_file, result_file):
        Output_Hmat(self, hmat_file)
        print("Wrote to " + hmat_file)
        Output_Vec(x, vec_file)
        print("Wrote to " + vec_file)
        cmd = "./matvec " + hmat_file + " " + vec_file + " " + result_file
        os.system(cmd)
        # Read in the result
        lines = [np.float64(line.rstrip('\n')) for line in open(result_file)]
        result = np.asarray(lines)
        return result

    def CallMatVec(self, x):
        hmat_file = "HMat.dat"
        vec_file = "vec.dat"
        result_file = "result.dat"
        Output_Hmat(self, hmat_file)
        Output_Vec(x, vec_file)
        cmd = "./matvec "  + hmat_file + " " + vec_file + " " + result_file
        os.system(cmd)
        lines = [np.float64(line.rstrip('\n')) for line in open(result_file)]
        result = np.asarray(lines)
        return result
        '''
        result = np.zeros(x.shape[0], dtype=np.float64)
        for low_rank in self.subMat:
            for i in xrange(0, low_rank.n):
                x_index = low_rank.start_i + i
                for j in xrange(0, low_rank.d):
                    for k in xrange(0, low_rank.k):
                        result[x_index] = result[x_index] + low_rank.Y[i, k] * low_rank.Z[k, j] * x[x_index]
        return result
        '''

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
        start_j = sub_mat.start_j
        line = str(start_i) + "\n" + str(start_j) + "\n" + str(n) + "\n" + str(k) + "\n" + str(m) + "\n"
        output.write(line)
        #print(sub_mat.Y.shape, sub_mat.Z.shape)
        for i in range(0, n):
            line = ""
            for j in range(0, k):
                line = line + str(sub_mat.Y[i][j]) + "\n"
            output.write(line)
        for j in range(0, m):
            line = ""
            for i in range(0, k):
                line = line + str(sub_mat.Z[i][j]) + "\n"
            output.write(line)
    output.close()

def Output_Vec(x, filename):
    output = open(filename, 'w')

    m = str(x.shape[0]) + "\n"
    output.write(m)
    for i in range(0, x.shape[0]):
        ele = str(x[i]) + "\n"
        output.write(ele)
    output.close()
