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
outdata = "initial_test.dat"
mat = Restructure(data)
print("Built Matrix")
min_tol = 0.5
max_tol = 1.0
tol_diff = .01
num_compute = int((max_tol + tol_diff - min_tol)/tol_diff)
bits_to_mb = 8 * 10e8
old_size = n * m
tol_arr = [] # Array of values for the tolerance
size_arr = [] # Array of values for the reduced matrix size
rel_err_arr = [] # Array of values for the relative error
stor_arr = [] # Total Storage in MB

index = 1
for tol in np.arange(min_tol, max_tol + tol_diff, tol_diff):
    print("Tolerance = " + str(tol) + " \t(" + str(index) + "/" + str(num_compute) + ")")
    A = mat.copy()
    low_rank, sz = RankSVD(A, bins, tol)
    new_size = np.sum(sz)
    size_arr.append(new_size)
    rel_err = FrobDiff(mat, low_rank)
    rel_err_arr.append(rel_err)
    tol_arr.append(tol)
    storage = (new_size * 64.0)/bits_to_mb
    stor_arr.append(storage)
    index = index + 1

print("Finished Computations")

# Write the data to a file
fout = open(outdata, 'w')

for i in range(0, len(tol_arr)):
    str_out = str(tol_arr[i]) + '\t' + str(size_arr[i]) + '\t' + str(rel_err_arr[i]) + '\t' + str(stor_arr[i]) + '\n'
    fout.write(str_out)

fout.close()
print("Wrote data to " + outdata)

# Plot the data
plt.plot(tol_arr, size_arr)
plt.xlabel("Tolerance")
plt.ylabel("Number of Matrix Elements")
plt.title("Matrix Size for a Given Tolerance")
plt.savefig("tol_size.pdf")

plt.clf()
plt.plot(tol_arr, rel_err_arr)
plt.xlabel("Tolerance")
plt.ylabel("Relative Error")
plt.title("Relative Error for a Given Tolerance")
plt.savefig("tol_error.pdf")

plt.clf()
plt.plot(size_arr, rel_err_arr)
plt.xlabel("Number of Matrix Elements")
plt.ylabel("Relative Error")
plt.title("Relative Error for Given Number of Matrix Elements")
plt.savefig("size_err.pdf")

plt.clf()
plt.plot(rel_err_arr, stor_arr)
plt.xlabel("Relative Error")
plt.ylabel("Matrix Storage (MB)")
plt.title("Storage Costs of Matrix (double)")
plt.savefig("storage_err.pdf")

print("Finished plots")
