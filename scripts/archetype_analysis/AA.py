import numpy as np
import h5py
from py_pcha import PCHA

with h5py.File('svd_40.hdf5', 'r') as f:
    u = f['/u'][:]    # shape (n, k)
    s = f['/s'][:]    # shape (k,)
    v = f['/v'][:]    # shape (k, m)

# # reformat s to function with matmul
# Sdiag = da.diag(s)

# # rebuild the original matrix X = u @ s @ v^T
# X = da.matmul(u, da.matmul(Sdiag, v.T))

X_reduced = np.diag(s) @ v

# sanity check: is X_reduced of dim(k, N)?
print(f"X_reduced shape: {X_reduced.shape}")

# run PCHA
n = 8
delta = 0.0 # inflation factor for the convex hull
XC, S_PCHA, C, SSE, varexpl = PCHA(X_reduced, noc=n, delta=delta)

print("Archetypal Analysis finished:")
print(" - XC shape:", XC.shape)
print(" - S shape:", S_PCHA.shape)
print(" - C shape:", C.shape)
print(" - SSE:", SSE)
print(" - Variance explained:", varexpl)

XC_full = u @ XC

print(f"XC_full shape: {XC_full.shape}")
"""
print("Is S equal to C.T?")
print("First 10 values of S:")
print(S_PCHA[:, :6])
print(np.sum(S_PCHA[:, :6], axis = 0))
print("First 10 values of C:")
print(C[:10, :])
print(C.T[:10, :])
print("Sum of C rows")
print(np.sum(C, axis=0))
print(np.where(S_PCHA.T == C))
"""
with h5py.File(f'pcha_results_{n}a_0d.hdf5', 'w') as f:
    # factor scores (reconstructed components)
    f.create_dataset('XC',      data=XC)
    # coefficients of each archetype, sums to 1 per example
    f.create_dataset('S_PCHA',  data=S_PCHA)
    # coefficients of examples to reconstruct archetypes, sums to 1 per archetype
    f.create_dataset('C',       data=C)
    # sum of squared errors (scalar or 1-D array)
    f.create_dataset('SSE',     data=SSE)
    # variance explained (scalar or 1-D array)
    f.create_dataset('varexpl', data=varexpl)