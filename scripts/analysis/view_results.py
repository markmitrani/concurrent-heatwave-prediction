import numpy as np
import h5py
import xarray as xr
import matplotlib.pyplot as plt

with h5py.File('data/pcha_results_8a.hdf5', 'r') as f: # run from mmi393 directory or gives error
    XC = f['/XC'][:]
with h5py.File('data/svd_40.hdf5', 'r') as f: # run from mmi393 directory or gives error
    u = f['/u'][:]

indexes = np.load("data/z_mapping.npz")
lon_idx = indexes["lon"]
lat_idx = indexes["lat"]

# Transformation from low rank SVD representation to full size
XC_full = u @ XC

nr_archetypes = XC_full.shape[1]
print(f"Processing visualization for {nr_archetypes} archetypes...")

XC_full_da = xr.DataArray(
  XC_full,
  dims   = ("z", "archetype"),
  coords = {
    "archetype": np.arange(1, XC_full.shape[1]+1),
    "lat":       ("z", lat_idx),
    "lon":       ("z", lon_idx),
  }
)

# unstack z -> lon,lat
XC_full_map = XC_full_da.set_index(z=("lon","lat")).unstack("z")
# expected dims: ("archetype", "lon", "lat")
print(XC_full_map.shape)
print(XC_full_map.dims)

# note: xarray infers x from dims[1], and y from dims[0].
# therefore a transpose is required:
XC_full_map = XC_full_map.transpose("archetype", "lat", "lon")
# now longitude will be plotted as x, and latitude as y.

# Plot in a grid using xarray's .plot for geospatial data
#fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False) # for a=8
if (nr_archetypes == 4):
    fig, axes = plt.subplots(4, 1, figsize=(16, 8), squeeze=False)
elif (nr_archetypes == 6):
    fig, axes = plt.subplots(6, 1, figsize=(16, 12), squeeze=False)
elif (nr_archetypes == 8):
    fig, axes = plt.subplots(8, 1, figsize=(16, 16), squeeze=False)
axes = axes.flatten()

for i, ax in enumerate(axes):
    arr = XC_full_map.isel(archetype=i)
    # Geospatial style with xarray
    im = arr.plot(
        ax=ax,
        cmap='coolwarm',
        add_colorbar=True,
        cbar_kwargs={"label": "Archetype Value"},
        robust=True
    )
    ax.set_title(f"Archetype {i+1}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.tight_layout()

# save figure(s)
fname = f"plots/archetypes_{nr_archetypes}.png"
fig.savefig(fname, dpi=300, bbox_inches='tight')
plt.close(fig)

"""
for i in range(S_PCHA_map.shape[0]):
    fig, ax = plt.subplots()
    arr = S_PCHA_map.isel(archetype=i).values
    im = ax.imshow(arr, origin='lower', aspect='auto')
    ax.set_title(f"Archetype {i+1}")
    fig.colorbar(im, ax=ax)
    
    fname = f"archetype_{i+1}.png"
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
"""