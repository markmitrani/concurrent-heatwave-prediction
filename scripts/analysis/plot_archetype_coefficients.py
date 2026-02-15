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

def plot_archetype_coefficients(S_PCHA, stream_ds, plots_dir):
    print(f"spcha size: {S_PCHA.shape}")
    print(f"stream_ds time size: {stream_ds.coords['time'].shape}")
    print(f"stream_ds time size: {stream_ds.coords['time']}")
    print(f"stream_ds unique time size: {np.unique(stream_ds.coords['time'].values).shape}")

    grouped_ds = stream_ds.groupby('time')
    time = stream_ds['time']
    
    dates_dayofyear = stream_ds.groupby('time.dayofyear').count().coords['dayofyear'].values
    dayofyear_coefs = []

    dates_dayandyear = stream_ds.groupby('time').count().coords['time'].values
    dayandyear_coefs = []

    for i in range(8):
        weights = S_PCHA.T[:, i] # shape: (time, n_arch) -> (time, 1)
        weight_da = xr.DataArray(weights, dims="time", coords={"time": time})
        
        means = weight_da.groupby('time.dayofyear').mean().values
        print(means.shape)
        dayofyear_coefs.append(means)

        means = weight_da.groupby('time').mean().values
        print(means.shape)
        dayandyear_coefs.append(means)
    
    fig, ax = plt.subplots()
    bottom = np.zeros(len(dayofyear_coefs[0]))
        
    for i, dayofyear_coef in enumerate(dayofyear_coefs):
        p = ax.bar(dates_dayofyear, dayofyear_coef, width=0.5, label=i, bottom=bottom)
        bottom += dayofyear_coef

    ax.set_title("Archetypes: Grouped by Day of Year")
    ax.legend(loc="upper right")

    fname = os.path.join(plots_dir, f"archetypes_dayofyear.png")
    fig.savefig(fname, dpi=450, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    bottom = np.zeros(len(dayandyear_coefs[0]))
        
    for i, dayandyear_coef in enumerate(dayandyear_coefs):
        p = ax.bar(dates_dayandyear, dayandyear_coef, width=0.5, label=i, bottom=bottom)
        bottom += dayandyear_coef

    ax.set_title("Archetypes: Grouped by Dates")
    ax.legend(loc="upper right")

    fname = os.path.join(plots_dir, f"archetypes_dayandyear.png")
    fig.savefig(fname, dpi=450, bbox_inches='tight')
    plt.close(fig)

    print(f"groupby datetime: {stream_ds.groupby('time')}")
    # print(f"groupby time of day: {stream_ds.groupby('time.dayofyear')['time'].month}")
def main():
    repo_dir = "concurrent-heatwave-prediction"

    style_file_path = os.path.join(repo_dir, "scripts/analysis/paper.mplstyle")
    plt.style.use(style_file_path)

    data_dir = os.path.join("concurrent-heatwave-prediction", "data")
    data_dir = os.path.join(data_dir, "lat30-60")

    plots_dir = os.path.join("concurrent-heatwave-prediction", "plots")
    plots_dir = os.path.join(plots_dir, "lat30-60")

    pcha_path = os.path.join(data_dir, 'pcha_results_8a_0d.hdf5')
    stream_path = os.path.join(data_dir, 'lentis_stream250_JJA_2deg_101_deseason_smsub_sqrtcosw_lat3060.nc')

    with h5py.File(os.path.join(pcha_path), 'r') as f: # run from mmi393 directory or gives error
        XC = f['/XC'][:]
        S_PCHA = f['/S_PCHA'][:]
    
    n_arch = XC.shape[1]
    n_samples = S_PCHA.shape[1]

    ds_str = xr.open_dataset(stream_path)
    ds_time = ds_str.coords['time']

    plot_archetype_coefficients(S_PCHA, ds_str, plots_dir)


if __name__ == '__main__':
    main()