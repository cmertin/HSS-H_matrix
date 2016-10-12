from mat import *
from compress_mat import *
import scipy.linalg.interpolative as sli


curve = "H"
mat_file = "mob_" + curve + ".bin"
print("Matrix File: " + mat_file)
n = int(3000)
m = int(3000)
data = ReadBinary(mat_file, n * m)
mat = Restructure(data)
print("Built Matrix")

Uk, sk, Vk = sli.svd(mat, 1e-8)

print(Uk.shape[0], Uk.shape[1], sk.shape[0], Vk.shape[0], Vk.shape[1])
