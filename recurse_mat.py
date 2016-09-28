from mat import *

def MatrixSplit(mat):
    n = mat.shape[0]
    m = mat.shape[1]
    n2_a = int(floor(n/2))# + (n % 2)
    n2_b = int(floor(n/2))
    m2_a = int(floor(m/2))# + (m % 2)
    m2_b = int(floor(m/2))
    mat_a1 = [min(n2_a, m2_a), n2_a, m2_a, 0, 0]
    mat_a2 = [min(n2_b, m2_a), n2_a, m2_b, 0, m2_a]
    mat_b1 = [min(n2_a, m2_b), n2_b, m2_a, n2_a, 0]
    mat_b2 = [min(n2_b, m2_b), n2_b, m2_b, n2_a, m2_a]
    ranks = [mat_a1, mat_a2, mat_b1, mat_b2]
    return ranks

def LowRank_Recurse(mat, old_rank, splits, ranks, tol, min_rank):
    max_rank = 20#floor(old_rank/6)
    for split in splits:
        n = split[1]
        m = split[2]
        start_i = split[3]
        start_j = split[4]
        sub_matrix = SubMatrix(mat, m, n, start_i, start_j)
        U, s, V = np.linalg.svd(sub_matrix, full_matrices = False)
        #print(Rank(s), old_rank)
        if Rank(s) <= min_rank:
            continue

        low_rank, new_rank = LowRankMat(U, s, V, tol)
        print(new_rank, old_rank)
        if new_rank <= max_rank:
            sub_data = [new_rank, n, m, start_i, start_j]
            ranks.append(sub_data)
        else:
            sub_splits = MatrixSplit(sub_matrix)
            LowRank_Recurse(sub_matrix, old_rank, sub_splits, ranks, tol, min_rank)
            low_rank = sub_matrix

        mat = UpdateMat(mat, low_rank, start_i, start_j)
    return mat, ranks

''''''
curve = "H"
level = "5"
n = 3000
m = 3000
mat_file = "mob_" + curve + ".bin"
bin_file = "level_" + level + "_" + curve + ".dat"
print("Matrix File: " + mat_file)
data = ReadBinary(mat_file, n * m)
x = np.random.rand(m)
bins = [int(1000/8)] * 8
outdata = "initial_test.dat"
mat = Restructure(data)
print("Built Matrix")

min_rank = 3
tol = 0.95
D = mat.copy()
#D = np.random.rand(100,100)#np.zeros((100, 100), dtype=np.float64)

print(D)
print('\n\n')
print(D.shape)
print('\n\n')

splits = MatrixSplit(D)

ranks = []

U, s, V = np.linalg.svd(D, full_matrices = True)

low_rank, ranks = LowRank_Recurse(D, Rank(s), splits, ranks, tol, min_rank)

print("\n\n")
print(low_rank)
print("\n\n")

num_elements = 0
for rank in ranks:
    num_elements = num_elements + 2 * rank[0] * rank[1]

print(num_elements)

rel_err = FrobDiff(mat, low_rank)
print(rel_err)