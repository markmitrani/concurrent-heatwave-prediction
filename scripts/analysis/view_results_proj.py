import argparse
import sys
import os
import numpy as np
import h5py
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def main(plot_anomalies):
    repo_dir = "concurrent-heatwave-prediction"
    data_dir = os.path.join("concurrent-heatwave-prediction", "data")
    data_dir = os.path.join(data_dir, "lat30-60")
    plots_dir = os.path.join("concurrent-heatwave-prediction", "plots")
    plots_dir = os.path.join(plots_dir, "lat30-60")

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

    data_min = float(XC_full_map.min())
    data_max = float(XC_full_map.max())
    absmax = max(abs(data_min), abs(data_max))
    vmin, vmax = -absmax, absmax

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
        fig, axes = plt.subplots(4, 2, figsize=(16, 8), subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=False
        )

    axes = axes.flatten()

    # tighten margins and vertical spacing
    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.96,
        bottom=0.12,
        hspace=0.15,
        wspace=0.075
    )

    center_colorbar = True

    for i, ax in enumerate(axes):
        arr = XC_full_map.isel(archetype=i)

        # lat_span = float(XC_full_map.lat.max() - XC_full_map.lat.min())
        # lon_span = float(XC_full_map.lon.max() - XC_full_map.lon.min())
        # orig_aspect = lat_span / lon_span
        ax.set_aspect("auto")

        if not center_colorbar:
            vmin=XC_full_map.min().item()
            vmax=XC_full_map.max().item()

        # pcolormesh plot with geographic transform
        pcm = arr.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            add_labels=False
        )

        # map features
        ax.coastlines(resolution='110m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        # ax.set_title(f"Archetype {i+1}")

        # gridlines
        gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5)
        
        show_left   = (i % 2 == 0)
        show_right  = not show_left
        show_top    = (i <= 1)
        show_bottom = (i >= 6)

        if i % 2 == 0:
            gl.left_labels = True
        else:
            gl.right_labels = True
        if i<=1:
            gl.top_labels = True
        elif i>=6:
            gl.bottom_labels = True

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # large subplot label in upper-left
        ax.text(
            0.02, 0.95,               # position in axes coords
            f"{i+1}",                 # 1, 2, 3, ...
            transform=ax.transAxes,
            fontsize=18,
            fontweight='bold',
            va='top', ha='left',
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )
    
    # cbar = fig.colorbar(pcm, ax=axes, orientation='horizontal', shrink=0.40, pad=0.08)
    cbar = fig.colorbar(pcm, ax=axes, orientation="horizontal", shrink=0.7, fraction=0.035, pad=0.08)

    cbar.set_label("stream250")

    # save figure(s)
    flag_string = "_anom" if plot_anomalies else ""
    fname = os.path.join(plots_dir ,f"archetypes_{nr_archetypes}_on_map_0d{flag_string}_v2.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Visualize AA Results')
    parser.add_argument('--plot_anomalies', action='store_true',
                        help="Subtract climatological mean from archetypes if True, default False")
    args = parser.parse_args()

    main(args.plot_anomalies)