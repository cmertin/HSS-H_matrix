from mat import *
from compress_mat import *
import scipy.linalg.interpolative as sli

def MatrixSplit(mat):
    n = mat.shape[0]
    m = mat.shape[1]
    n2_a = int(floor(n/2)) + (n % 2)
    n2_b = int(floor(n/2)) - (n % 2)
    m2_a = int(floor(m/2)) + (m % 2)
    m2_b = int(floor(m/2)) - (m % 2)
    mat_a1 = [min(n2_a, m2_a), n2_a, m2_a, 0, 0]
    mat_a2 = [min(n2_b, m2_a), n2_a, m2_b, 0, m2_a]
    mat_b1 = [min(n2_a, m2_b), n2_b, m2_a, n2_a, 0]
    mat_b2 = [min(n2_b, m2_b), n2_b, m2_b, n2_a, m2_a]
    splits = [mat_a1, mat_a2, mat_b1, mat_b2]
    return splits

def LowRank_Recurse(mat, Hmat, max_rank, splits, tol, min_rank, level = 1):
    level = level + 1
    for split in splits:
        n = split[1]
        m = split[2]
        start_i = split[3]
        start_j = split[4]
        sub_matrix = SubMatrix(mat, n, m, start_i, start_j)
        U, s, V = np.linalg.svd(sub_matrix, full_matrices = False)
        local_rank = Rank(s)
        if local_rank <= min_rank:
            Hmat.add_lowrank(sub_matrix, local_rank, start_i, start_j)
            continue

        low_rank, new_rank = LowRankMat(U, s, V, tol)

        if new_rank == local_rank:
            Hmat.add_lowrank(sub_matrix, local_rank, start_i, start_j)
            continue

        if new_rank < (max_rank/(2**level)):
            Hmat.add_lowrank(low_rank, new_rank, start_i, start_j)
        else:
            old_level = level
            local_rank = new_rank
            sub_splits = MatrixSplit(sub_matrix)
            LowRank_Recurse(sub_matrix, Hmat, local_rank, sub_splits, tol, min_rank, level)
            level = old_level
            low_rank = sub_matrix

    return Hmat

curve = "H"
level = "5"

mat_file = "mob_" + curve + ".bin"
bin_file = "level_" + level + "_" + curve + ".dat"
print("Matrix File: " + mat_file)
n = int(3000)
m = int(3000)
data = ReadBinary(mat_file, n * m)
x = np.random.rand(m)
outdata = "initial_test.dat"
mat = Restructure(data)

min_rank = 16#500
min_tol = 0.8
tol_diff = 0.01
max_tol = 1.0 + tol_diff

Hmat = HMat(3000, 3000)

nm = [1000]#[1000, 2000, 3000, 4000, 6000, 10000, 20000]
for n in nm:
    m = n
    outdata = str(n) + "_" + str(m) + "_data.dat"
    #mat = np.random.rand(n,m)
    print("Built Matrix")
    U, s, V = np.linalg.svd(mat, full_matrices = True)
    print("Rank = " + str(Rank(s)))
    tol_arr = []
    rel_err_arr = []
    num_elements_arr = []

    #for tol in np.arange(min_tol, max_tol, tol_diff):
    tol = .95
    print("Tolerance = " + str(tol))
    D = mat.copy()
    splits = MatrixSplit(D)
    ranks = []
    U, s, V = np.linalg.svd(D, full_matrices = True)
    Hmat2 = LowRank_Recurse(D, Hmat, Rank(s), splits, tol, min_rank)

    sub_matrices = Hmat.GetSubMat()
    num_elements = 0
    num_k = 0
    zero_count = 0
    for sub_matrix in sub_matrices:
        n1 = sub_matrix.n
        m1 = sub_matrix.d
        k1 = sub_matrix.k
        num_elements = num_elements + n1 * m1
        num_k = num_k + 2 * n1 * k1
    print(num_elements, num_k)

    print("Sub Matrices: " + str(len(sub_matrices)))

    r1 = np.dot(mat, x)
    r2 = Hmat.MatVec(x)
    rel_err = FrobDiff(r1, r2)
    print(rel_err)
'''
        rel_err = FrobDiff(mat, low_rank)

        tol_arr.append(tol)
        rel_err_arr.append(rel_err)
        num_elements_arr.append(num_elements)
        print('\t' + str(rel_err))

    fout = open(outdata, 'w')

    for i in range(0, len(tol_arr)):
        str_out = str(tol_arr[i]) + '\t' + str(rel_err_arr[i]) + '\t' + str(num_elements_arr[i]) + '\n'
        fout.write(str_out)

    fout.close()
    print("Wrote data to " + outdata)

    plot_title_tol = "Tolerance vs Number of Elements [" + str(n) + " x " + str(m) + "]"
    plot_title_err = "Relative Error vs Number of Elements [" + str(n) + " x " + str(m) + "]"
    file_tol = str(n) + "_" + str(m) + "_tol_elements.pdf"
    file_err = str(n) + "_" + str(m) + "_rel-err_elements.pdf"

    plt.clf()
    plt.plot(tol_arr, num_elements_arr)
    plt.xlabel("Tolerance")
    plt.ylabel("Number of Matrix Elements")
    plt.title(plot_title_tol)
    #plt.ticklabel_format(style="sci", axis='y', scilimits=(6,6))
    plt.savefig(file_tol, format="pdf", bbox_inches="tight")

    plt.clf()
    plt.plot(rel_err_arr, num_elements_arr)
    plt.xlabel("Relative Error")
    plt.ylabel("Number of Matrix Elements")
    plt.title(plot_title_err)
    plt.savefig(file_err, format="pdf", bbox_inches="tight")

    print("Finished Plotting")
    '''
print("DONE")
