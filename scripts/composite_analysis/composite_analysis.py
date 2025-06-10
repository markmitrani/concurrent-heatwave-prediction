import os
import numpy as np
import h5py
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import argparse

def compute_composites(dataset_tas, S_PCHA, method='argmax'):
    tas = dataset_tas['tas']
    time = dataset_tas['time']
    n_arch = S_PCHA.shape[0]

    if method == 'argmax':
        arch_indices = np.argmax(S_PCHA, axis=0)
        arch_da = xr.DataArray(arch_indices, dims="time", coords={"time": time})
        tas_labeled = dataset_tas.assign(archetype=arch_da)
        grouped = tas_labeled.groupby('archetype').mean('time')
        return grouped['tas']

    elif method == 'weighted':
        composites = []

        for k in range(n_arch):
                weights = S_PCHA.T[:, k] # shape: (time, n_arch) -> (time, 1)
                weight_da = xr.DataArray(weights, dims="time", coords={"time": time})
                weighted_tas = tas * weight_da
                composite_k = weighted_tas.sum(dim="time") / weight_da.sum()
                composites.append(composite_k)

        return xr.concat(composites, dim="archetype")

    else:
        raise ValueError(f"Unknown method: {method}")

def plot_composites(composite_da, plots_dir, method, varname='tas'):
    n_arch = composite_da.sizes['archetype']

    fig_height = {4: 12, 6: 18, 8: 24}.get(n_arch, 4 * n_arch)
    fig, axes = plt.subplots(n_arch, 1, figsize=(16, fig_height),
                            subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=False)
    axes = axes.flatten()

    vmin = composite_da.min().item()
    vmax = composite_da.max().item()

    for i, ax in enumerate(axes):
        arr = composite_da.isel(archetype=i)

        # pcolormesh plot with geographic transform
        pcm = arr.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='coolwarm',
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
            cbar_kwargs={"label": varname}
        )
      
      # map features & gridlines
        ax.coastlines(resolution='110m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.set_title(f"Composite for Archetype {i+1}")
        ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fname = os.path.join(plots_dir, f"composite_{n_arch}_arch_{method}.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main(method):
    repo_dir = "concurrent-heatwave-prediction"
    data_dir = os.path.join(repo_dir, "data", "lat30-60")
    plots_dir = os.path.join(repo_dir, "plots", "lat30-60")

    tas_path = os.path.join(data_dir, 'lentis_tas_JJA_2deg_101_deseason.nc')
    pcha_path = os.path.join(data_dir, 'pcha_results_8a.hdf5')

    dataset_tas = xr.open_dataset(tas_path)

    with h5py.File(pcha_path, 'r') as f:
        S_PCHA = f['/S_PCHA'][:]

    composite_da = compute_composites(dataset_tas, S_PCHA, method=method)
    plot_composites(composite_da, plots_dir, method, varname='tas')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and visualize TAS composite analysis conditioned on stream250 archetypes.')
    parser.add_argument('--method', choices=['argmax', 'weighted'],
                        help='Composite analysis method: "argmax" for hard grouping, "weighted" for soft composite')
    args = parser.parse_args()

    main(args.method)