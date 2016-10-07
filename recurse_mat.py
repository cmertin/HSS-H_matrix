from mat import *
import scipy as sc

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

def LowRank_Recurse(mat, max_rank, splits, ranks, tol, min_rank, level):
    level = level + 1
    for split in splits:
        n = split[1]
        m = split[2]
        start_i = split[3]
        start_j = split[4]
        sub_matrix = SubMatrix(mat, m, n, start_i, start_j)
        U, s, V = np.linalg.svd(sub_matrix, full_matrices = False)
        #print(Rank(s), old_rank)
        local_rank = Rank(s)
        if local_rank <= min_rank:
            continue

        low_rank, new_rank = LowRankMat(U, s, V, tol)
        #print(new_rank, old_rank)
        if new_rank <= (max_rank/(2**level)):
            sub_data = [new_rank, n, m, start_i, start_j]
            ranks.append(sub_data)
        else:
            old_level = level
            local_rank = new_rank
            sub_splits = MatrixSplit(sub_matrix)
            LowRank_Recurse(sub_matrix, local_rank, sub_splits, ranks, tol, min_rank, level)
            level = old_level
            low_rank = sub_matrix

        mat = UpdateMat(mat, low_rank, start_i, start_j)
    return mat, ranks

curve = "H"
level = "5"

mat_file = "mob_" + curve + ".bin"
bin_file = "level_" + level + "_" + curve + ".dat"
print("Matrix File: " + mat_file)
n = 3000
m = 3000
data = ReadBinary(mat_file, n * m)
x = np.random.rand(m)
outdata = "initial_test.dat"
mat = Restructure(data)

min_rank = 3
min_tol = 0.8
tol_diff = 0.01
max_tol = 1.0 + tol_diff


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

    for tol in np.arange(min_tol, max_tol, tol_diff):
        print("Tolerance = " + str(tol))
        D = mat.copy()
        splits = MatrixSplit(D)
        ranks = []
        U, s, V = np.linalg.svd(D, full_matrices = True)
        low_rank, ranks = LowRank_Recurse(D, Rank(s), splits, ranks, tol, min_rank, 1)

        num_elements = 0
        nm_ = 0
        for rank in ranks:
            num_elements = num_elements + 2 * rank[0] * rank[1]
            nm_ = nm_ + rank[1] * rank[2]

        nm_diff = 3000*3000 - nm_
        print("Diff: " + str(nm_diff))
        num_elements = num_elements + nm_diff
        print(num_elements)

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
print("DONE")
