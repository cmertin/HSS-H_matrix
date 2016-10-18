from mat import *
from compress_mat import *
import scipy.linalg.interpolative as sli

def MatrixCheck(Hmat):
    mat = np.zeros((Hmat.n, Hmat.m))
    count = 0
    for submat in Hmat.subMat:
        start_i = submat.start_i
        start_j = submat.start_j
        n = submat.n
        m = submat.d
        k = submat.k
        count = count + n * m
        for i in range(0, n):
            for j in range(0, m):
                row = start_i + i
                col = start_j + j
                #print(row, col)
                mat[row,col] = 1
    nnz = NNZ(mat)
    print(mat)
    print(count, nnz)

def MatrixSplit(n, m, start_i = 0, start_j = 0):
    n2_a = int(floor(n/2)) + (n % 2)
    n2_b = int(floor(n/2))# - (n % 2)
    m2_a = int(floor(m/2)) + (m % 2)
    m2_b = int(floor(m/2))# - (m % 2)
    mat_a1 = [n2_a, m2_a, start_i, start_j]
    mat_a2 = [n2_a, m2_b, start_i, start_j + m2_a]
    mat_b1 = [n2_b, m2_a, start_i + n2_a, start_j]
    mat_b2 = [n2_b, m2_b, start_i + n2_a, start_j + m2_a]
    splits = [mat_a1, mat_a2, mat_b1, mat_b2]
    return splits

def LowRank_Recurse(mat, Hmat, splits, tol, min_rank):
    #print(splits)
    for split in splits:
        n = split[0]
        m = split[1]
        start_i = split[2]
        start_j = split[3]

        sub_matrix = SubMatrix(mat, n, m, start_i, start_j)
        U, s, V = np.linalg.svd(sub_matrix, full_matrices = False)
        local_rank = Rank(s)
        if local_rank <= min_rank:
            #print(split)
            Hmat.add_lowrank(sub_matrix, local_rank, start_i, start_j, True)
            continue

        low_rank, new_rank = LowRankMat(U, s, V, tol)
        #Uk, sk, Vk = sli.svd(sub_matrix, tol)
        #new_rank = Rank(sk)
        #low_rank, new_rank = CompressMatrixID(Uk, sk, Vk, start_i, start_j)

        if new_rank < min(n/2, m/2):#(max_rank/(2**level)):
            Hmat.add_lowrank(sub_matrix, new_rank, start_i, start_j)
            #print('\t', split, new_rank)
            continue
        else:
            local_rank = new_rank
            sub_splits = MatrixSplit(n, m, start_i, start_j)
            LowRank_Recurse(mat, Hmat, sub_splits, tol, min_rank)
            low_rank = sub_matrix

    return Hmat

curve = "H-1000"
level = "5"

mat_file = "mob_" + curve + ".bin"
bin_file = "level_" + level + "_" + curve + ".dat"
print("Matrix File: " + mat_file)
n = int(3000)
m = int(3000)
data = ReadBinary(mat_file, n * m)
x = np.random.rand(m)

for i in range(0, m):
    x[i] = x[i] * 100

outdata = "initial_test.dat"
hmat_file = "HMat.dat"
vec_file = "vec.dat"
result_file = "result.dat"
mat = Restructure(data)

min_rank = 16#500#16
min_tol = 0.8
tol_diff = 0.01
max_tol = 1.0 + tol_diff

Hmat = HMat(3000, 3000)

print("Built Matrix")
U, s, V = np.linalg.svd(mat, full_matrices = True)
print("Rank = " + str(Rank(s)))

tol = .99
print("Tolerance = " + str(tol))
D = mat.copy()
splits = MatrixSplit(n, m)
#print(splits)
#U, s, V = np.linalg.svd(D, full_matrices = True)
Hmat2 = LowRank_Recurse(D, Hmat, splits, tol, min_rank)

sub_matrices = Hmat.GetSubMat()
num_elements = 0
num_k = 0
zero_count = 0
for sub_matrix in sub_matrices:
    n1 = sub_matrix.n
    m1 = sub_matrix.d
    k1 = sub_matrix.k
    num_elements = num_elements + n1 * m1
    if n1 == k1 and m1 == k1:
        num_k = num_k + n1 * m1
    else:
        num_k = num_k + 2 * n1 * k1

print(num_elements, num_k)

#MatrixCheck(Hmat)

print("Sub Matrices: " + str(len(sub_matrices)))

r1 = np.dot(mat, x)
r2 = Hmat.MatVec(x, hmat_file, vec_file, result_file)
rel_err = FrobDiff(r1, r2)
print(rel_err)
print("DONE")
