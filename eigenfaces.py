import pandas as pd
import numpy as np
import cupy as cp
from sklearn.datasets import fetch_lfw_people

x = cp.arange(6).reshape(2,3)
print(x)

y = x.sum(axis=1)
print(y)

import sys
sys.exit()

# Read dataset
lfw = fetch_lfw_people(data_home='./data_lfw/', resize=1.0, min_faces_per_person=1, slice_=None)

# Spec of dataset
nimages, img_height, img_width = lfw.images.shape
print("Nunmber of images =", img_width)
print("Image size = {} x {}".format(img_height, img_width))

df = pd.DataFrame(lfw.data)

X = np.matrix(np.zeros([img_width * img_height, nimages]))
for idx in range(nimages):
    X[:, idx] = np.matrix(df.loc[idx, :].values).T

# Compute average face
print("start computing average face...")
Xavg = np.mean(X, axis=1)

# Compute the difference image (the centered data matrix)
print("start computing the difference image...")
Xc = X.copy() - Xavg @ np.ones([1, nimages])
Xc_gpu = cp.asarray(np.array(Xc))

# Compute the covariance matrix
print("start computing the covariance matrix...")
covariance_matrix = cp.dot(Xc_gpu, Xc_gpu.T) / nimages

# Compute the eigenvectors of the covariance matrix
print("start computing the eigenvectors of the covariance matrix...")
eigenvalues, eigenvectors = cp.linalg.eigh(covariance_matrix)

# sorting eigen vectors according to their corresponding eigen values
positions = eigenvalues.argsort()[::-1]
sorted_eigenvalues = (eigenvalues[positions])
sorted_eigenvectors = (eigenvectors[:, positions])
print("Largest eigenvalue  =", sorted_eigenvalues[0])
print("Smallest eigenvalue =", sorted_eigenvalues[-1])

# Save binary file
size = int(h * w)
covariance_matrix = covariance_matrix.astype(np.float32)
with open("data/covarianceMatrix_lfw_250x250", "wb") as f:
    f.write(size.to_bytes(4, 'little'))
    f.write(covariance_matrix.tobytes())