import argparse
import sys
import os
import numpy as np
import h5py
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

def main(plot_anomalies):
    repo_dir = "concurrent-heatwave-prediction"
    data_dir = os.path.join("concurrent-heatwave-prediction", "data")
    data_dir = os.path.join(data_dir, "deseason_smsub_sqrtcosw")
    plots_dir = os.path.join("concurrent-heatwave-prediction", "plots")
    plots_dir = os.path.join(plots_dir, "deseason_smsub_sqrtcosw")

    with h5py.File(os.path.join(data_dir, 'pcha_results_8a_0d.hdf5'), 'r') as f: # run from mmi393 directory or gives error
        XC = f['/XC'][:]
    with h5py.File(os.path.join(data_dir, 'svd_40.hdf5'), 'r') as f: # run from mmi393 directory or gives error
        u = f['/u'][:]

    indexes = np.load(os.path.join(data_dir, 'z_mapping.npz'))
    lon_idx = indexes["lon"]
    lat_idx = indexes["lat"]

    # Only used if plotting anomalies
    lentis_path = os.path.join(data_dir, 'lentis_stream250_JJA_2deg_101_deseason_smsubd_sqrtcosw_lat3060.nc')

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

    # some testing
    print(f"XC full (np) min and max: {XC_full.min()}, {XC_full.max()}")
    print(f"XC full map min and max: {XC_full_map.min().item()}, {XC_full_map.max().item()}")

    # note: xarray infers x from dims[1], and y from dims[0].
    # therefore a transpose is required:
    XC_full_map = XC_full_map.transpose("archetype", "lat", "lon")
    # now longitude will be plotted as x, and latitude as y.

    # optionally, subtract climatological mean
    if (plot_anomalies):
        # Open dataset and select the streamfunction variable (adjust name as needed)
        dataset = xr.open_dataset(lentis_path)
        streamfunc = dataset['stream']

        # Compute temporal mean over time dimension -> shape (lat, lon)
        climatology = streamfunc.mean(dim='time')
        print(climatology.dims)
        print(climatology.shape)

        for i in range(nr_archetypes):
            XC_full_map.isel(archetype=i)[:] = climatology.squeeze()

        print(f"Shape before: {XC_full_map.shape}")
        # Subtract from each archetype
        XC_full_map = (XC_full_map + climatology).squeeze()
        print(f"Shape after: {XC_full_map.shape}")

    # set up grid based on archetype number
    if (nr_archetypes == 4):
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=False)
    elif (nr_archetypes == 6):
        fig, axes = plt.subplots(6, 1, figsize=(16, 18), subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=False)
    elif (nr_archetypes == 8):
        fig, axes = plt.subplots(8, 1, figsize=(16, 24), subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=False)
    axes = axes.flatten()

    center_colorbar = False

    for i, ax in enumerate(axes):
        arr = XC_full_map.isel(archetype=i)

        if center_colorbar:
            vmin_raw = XC_full_map.min()
            vmax_raw = XC_full_map.max()
            absmax = max(abs(vmin_raw), abs(vmax_raw))
            vmin, vmax = -absmax, absmax
        else:
            vmin=XC_full_map.min().item()
            vmax=XC_full_map.max().item()

        # pcolormesh plot with geographic transform
        pcm = arr.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='coolwarm',
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True,
            cbar_kwargs={"label": "stream250"}
        )

        # map features
        ax.coastlines(resolution='110m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.set_title(f"Archetype {i+1}")

        # gridlines, optional
        ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # save figure(s)
    flag_string = "_anom" if plot_anomalies else ""
    fname = os.path.join(plots_dir ,f"archetypes_{nr_archetypes}_on_map{flag_string}.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Visualize AA Results')
    parser.add_argument('--plot_anomalies', action='store_true',
                        help="Subtract climatological mean from archetypes if True, default False")
    args = parser.parse_args()

    main(args.plot_anomalies)