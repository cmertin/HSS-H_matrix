from mat import *

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
mat = Restructure(data)
print("Built Matrix")
mat_b = mat.copy()
tol = .99
print("Tolerance = " + str(tol))

zeroDiag, diag = ZeroDiagonals(mat_b)
print("Built M and diag")

low_rank, rank = RankSVD(zeroDiag, bins, tol)
#print(rank)
totalMat = low_rank + diag
rel_err = FrobDiff(mat, totalMat)
print(rel_err)

print("DONE")
