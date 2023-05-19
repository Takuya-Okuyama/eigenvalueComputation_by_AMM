import numpy as np
from numpy import linalg as LA

# data = np.loadtxt("matrix.dat", delimiter=",")
data = np.fromfile("matrix.dat", dtype=np.float32)
nelements = len(data)
size = int(np.sqrt(nelements))

A = np.matrix(data.reshape((size, size)).astype(np.float64))
w, v = LA.eigh(A)

print("Largest eigenvalue  = {0:20f}".format(max(w)))
print("Smallest eigenvalue = {0:20f}".format(min(w)))