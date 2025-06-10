import dask.array as da
import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

def plot_svd(k, explained_variance_ratio, sse):
    k_vals = np.arange(1, k+1)

    # Plotting setup
    fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # Style settings
    blue = '#1f77b4'  # Matplotlib's default blue — same as the plot you posted
    marker_style = dict(marker='o', markersize=6, markerfacecolor='none', markeredgewidth=1.5)

    # Top panel for Explained Variance
    axs[0].plot(k_vals, explained_variance_ratio, color=blue, **marker_style, label='SVD')
    axs[0].set_ylabel("Explained variance")
    #axs[0].set_ylim(0.9, 1.01)
    axs[0].legend(frameon=False)
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].grid(False)

    # Bottom panel for SSE
    axs[1].plot(k_vals, sse, color=blue, **marker_style, label='SVD')
    axs[1].set_xlabel("PC number")
    axs[1].set_ylabel("Sum of squares error")

    # Formatting y-axis as scientific notation ×10^x
    axs[1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axs[1].legend(frameon=False)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].grid(False)

    plt.tight_layout()
    fig.savefig(f"svd_plot_{k}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

filename = "lentis_stream250_JJA_2deg_101_deseason_smsubd_sqrtcosw_lat3060.nc"

# read nc file
ds = xr.open_dataset(filename, engine="netcdf4", chunks='auto')

# check shapes before operation
print(ds["stream"].data.shape)

# TODO: Does lon/lat ordering matter in stacking?
ds = ds.stack(z=("lon", "lat"))

# retain z index to recover lon x lat structure later
z_index = ds["stream"].indexes["z"]

# Break it out into two arrays:
lon_idx = z_index.get_level_values("lon").to_numpy()
lat_idx = z_index.get_level_values("lat").to_numpy()

# save mapping (z <-> (lon, lat)) to file:
np.savez("z_mapping.npz", lon=lon_idx, lat=lat_idx)

X = ds["stream"].squeeze().data.T

# check final shape before SVD
print("Note: Make sure shape dimensions are ordered as (D, N)")
print("D = features per example")
print("N = number of examples")
print(f"SVD will be performed on: {X.shape}")

# perform SVD
k = 40
u, s, v = da.linalg.svd_compressed(X, k, compute=True)

# check SVD outputs' shapes
print("=== SVD Complete ===")
print(" Shapes:")
print(f"\tu: {u.shape}, s: {s.shape}, v: {v.shape}")

# 1. Total variance = sum of squared singular values divided by (n - 1)
explained_variance = (s ** 2) / (X.shape[1] - 1)  # shape: (k,)
total_variance = explained_variance.sum()

# 2. Explained variance ratio (fraction of total variance)
explained_variance_ratio = explained_variance / total_variance

# 3. SSE for rank-k approximations (sum of squared errors)
sse = np.cumsum(explained_variance[::-1])[::-1]

explvar = explained_variance.compute()
evr = explained_variance_ratio.compute()
evr_cumulative = np.cumsum(evr)
sse = sse.compute()
total = total_variance.compute()

# output expl_var and SSE 
print(f"Explained variance: {explvar}")
print(f"Total variance: {total}")
print(f"Explained variance ratio: {evr}")
print(f"Cumulative explained variance ratio: {evr_cumulative}")
print(f"SSE: {sse}")

# plot results
plot_svd(k, evr, sse)

# save u, s, v to disk
da.to_hdf5(f'svd_{k}.hdf5', {'/u': u, '/s': s, '/v': v})