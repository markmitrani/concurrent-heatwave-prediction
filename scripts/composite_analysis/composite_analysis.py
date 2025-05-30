import numpy as np
import h5py
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# import both nc's
stream_path = "data/lentis_stream250_JJA_2deg_101_deseason_spatialsub.nc"
dataset_stream = xr.open_dataset(stream_path)
# dataset_stream.drop_dims("plev")

tas_path = "data/lentis_tas_JJA_2deg_101_deseason.nc"
dataset_tas = xr.open_dataset(tas_path)

# sanity check part 1: these results should be the same in part 2
print(dataset_tas.isel(time=123)['tas'].isel(lon=0, lat=0).values)
print(dataset_tas.isel(time=74)['tas'].isel(lon=4, lat=8).values)

# join the nc's together
dataset_comb = dataset_stream.assign(tas=dataset_tas['tas'])

# sanity check part 2
print(dataset_comb.isel(time=123)['tas'].isel(lon=0, lat=0).values)
print(dataset_comb.isel(time=74)['tas'].isel(lon=4, lat=8).values)

# get S_PCHA from archetypes file
with h5py.File('data/pcha_results_8a.hdf5', 'r') as f: # run from mmi393 directory or gives error
        S_PCHA = f['/S_PCHA'][:]


# group indices based on whichever archetype is maximum there
arch_indices = np.argmax(S_PCHA, axis=0)

# sanity check 1
print(arch_indices[0], arch_indices[5], arch_indices[6], arch_indices[9119])

arch_da = xr.DataArray(arch_indices, dims="time", coords={"time": dataset_comb.time})
# sanity check 2 
print(arch_da.isel(time=0).values, arch_da.isel(time=5).values, arch_da.isel(time=6).values, arch_da.isel(time=9119).values)

# calculate the mean for each archetype's group
dataset_comb_labeled = dataset_comb.assign(archetype=arch_da)

# sanity check 3
print(dataset_comb_labeled.isel(time=0)['archetype'].values,
      dataset_comb_labeled.isel(time=5)['archetype'].values,
      dataset_comb_labeled.isel(time=6)['archetype'].values,
      dataset_comb_labeled.isel(time=9119)['archetype'].values)

# plot results (plus: define and show ROI)
grouped = dataset_comb_labeled.groupby('archetype').mean('time')

print(grouped['tas'].shape)

n_arch = S_PCHA.shape[0]

# set up grid based on archetype number
if (n_arch == 4):
      fig, axes = plt.subplots(4, 1, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()},
      constrained_layout=False)
elif (n_arch == 6):
      fig, axes = plt.subplots(6, 1, figsize=(16, 18), subplot_kw={'projection': ccrs.PlateCarree()},
      constrained_layout=False)
elif (n_arch == 8):
      fig, axes = plt.subplots(8, 1, figsize=(16, 24), subplot_kw={'projection': ccrs.PlateCarree()},
      constrained_layout=False)
axes = axes.flatten()

for i, ax in enumerate(axes):
      arr = grouped.isel(archetype=i)['tas']

      # pcolormesh plot with geographic transform
      pcm = arr.plot.pcolormesh(
      ax=ax,
      transform=ccrs.PlateCarree(),
      cmap='coolwarm',
      vmin=grouped['tas'].min().item(),
      vmax=grouped['tas'].max().item(),
      add_colorbar=True,
      cbar_kwargs={"label": "tas"}
      )

      # map features
      ax.coastlines(resolution='110m', linewidth=0.8)
      ax.add_feature(cfeature.BORDERS, linewidth=0.4)
      ax.set_title(f"Composite for Archetype {i+1}")

      # gridlines, optional
      ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5)
      ax.set_xlabel("Longitude")
      ax.set_ylabel("Latitude")

# save figure(s)
fname = f"plots/composite_{n_arch}_arch.png"
fig.savefig(fname, dpi=300, bbox_inches='tight')
plt.close(fig)