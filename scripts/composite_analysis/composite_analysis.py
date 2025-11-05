import os
import numpy as np
import h5py
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse

def build_box_mask_da(lat1: float, lat2: float, lon1: float, lon2: float, like_dataset: xr.DataArray) -> xr.DataArray:
    """
    Create a rectangular mask for a given region bounded by lat/lon ranges.
    """
    # check ordering
    if lat1 >= lat2:
        raise ValueError(f"lat1 must be < lat2 (got lat1={lat1}, lat2={lat2})")
    if lon1 >= lon2:
        raise ValueError(f"lon1 must be < lon2 (got lon1={lon1}, lon2={lon2})")
    
    lats = like_dataset['lat']
    lons = like_dataset['lon']

    lat_mask = (lats >= lat1) & (lats <= lat2)
    lon_mask = (lons >= lon1) & (lons <= lon2)

    # Convert to numpy arrays before broadcasting
    lat_mask_np = lat_mask.values[:, None]  # shape (n_lat, 1)
    lon_mask_np = lon_mask.values[None, :]  # shape (1, n_lon)

    mask_np = lat_mask_np & lon_mask_np  # shape (n_lat, n_lon)

    # wrap back into DataArray
    mask_da = xr.DataArray(
        mask_np.astype(np.int8),
        coords={'lat': lats, 'lon': lons},
        dims=['lat', 'lon'],
        name='mask'
    )

    return mask_da

def combine_masks(mask1: xr.DataArray, mask2: xr.DataArray, method: str = 'union') -> xr.DataArray:
    """
    Combine two masks using logical union or intersection.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Masks must have the same shape.")
    if not all(mask1.coords[dim].equals(mask2.coords[dim]) for dim in ['lat', 'lon']):
        raise ValueError("Masks must have matching coordinates.")
    
    if method == 'union':
        combined = (mask1.astype(bool) | mask2.astype(bool)).astype(np.int8)
    elif method == 'intersection':
        combined = (mask1.astype(bool) & mask2.astype(bool)).astype(np.int8)
    else:
        raise ValueError("Method must be 'union' or 'intersection'.")

    return xr.DataArray(
        combined,
        coords=mask1.coords,
        dims=mask1.dims,
        name='combined_mask'
    )

def score_archetypes_in_roi_debug(dataset_tas: xr.Dataset, arch_weights_np: np.ndarray,
                                  roi_mask_da: xr.DataArray = None, normalize_weights: bool = False, method = 'argmax') -> xr.Dataset:
    print("\n>>> Starting score_archetypes_in_roi_debug")

    # Validate shape
    n_arch, n_time = arch_weights_np.shape
    time_coords = dataset_tas['time']
    print(f"- arch_weights_np shape: {arch_weights_np.shape}")
    print(f"- dataset_tas time dim: {time_coords.shape}")

    if n_time != time_coords.size:
        raise ValueError(f"arch_weights_np has {n_time} time steps, but dataset has {time_coords.size}.")
    
    if method == 'weighted':
        w = xr.DataArray(
            arch_weights_np,
            dims=['archetype', 'time'],
            coords={'archetype': np.arange(n_arch), 'time': time_coords}
        )
    elif method == 'argmax':
        arch_indices = np.argmax(arch_weights_np, axis=0)  # shape: (time,)
        one_hot = np.zeros_like(arch_weights_np)           # shape: (archetype, time)
        one_hot[arch_indices, np.arange(n_time)] = 1       # set winners to 1

        w = xr.DataArray(
            one_hot,
            dims=['archetype', 'time'],
            coords={'archetype': np.arange(n_arch), 'time': time_coords}
        )
    
    # Convert weights to DataArray
    print(f"- w (weights) DataArray:\n  dims: {w.dims}, shape: {w.shape}\n  sample:\n{w.isel(archetype=0).values[:5]}")

    if normalize_weights:
        w = w.clip(min=0)
        w = w / w.sum('time')
        print(f"- weights normalized")

    tas_mean = dataset_tas.mean('time')
    tas_std = dataset_tas.std('time')
    print(f"- tas_mean, tas_std computed. Mean sample:\n{tas_mean['tas'].isel(lat=0, lon=0).values}")
    print(f"- tas_std sample:\n{tas_std['tas'].isel(lat=0, lon=0).values}")

    if roi_mask_da is None:
        roi_mask_da = xr.ones_like(dataset_tas.isel(time=0))
        print("- No ROI mask provided; using full domain")

    z = (dataset_tas - tas_mean) #/ tas_std
    print(f"- Z-scores computed. Sample:\n{z['tas'].isel(time=0, lat=0, lon=0).values}")

    print(np.allclose(z['tas'].lat.values, roi_mask_da.lat.values))  # Should be True
    print(np.allclose(z['tas'].lon.values, roi_mask_da.lon.values))  # Should be True
    roi_mask_da_aligned = roi_mask_da.reindex_like(z['tas'].isel(time=0), method='nearest')
    z_roi = z.where(roi_mask_da == 1)
    print(f"- ROI mask applied. Sample:\n{z_roi['tas'].isel(time=0).sel(lat=45, lon=-105, method='nearest').values}")

    # Broadcast weights
    z_roi = z_roi.expand_dims({'archetype': w.sizes['archetype']})
    z_roi = z_roi.transpose('archetype', 'lat', 'lon', 'time', ...)
    print(f"- z_roi shape after expanding: {z_roi['tas'].shape}")

    w_expanded = w.expand_dims({'lat': z_roi.dims['lat'], 'lon': z_roi.dims['lon']}).transpose('archetype', 'lat', 'lon', 'time')  # (archetype, 1, 1, time)
    print(f"- w_expanded shape: {w_expanded.shape}")

    zw_roi = z_roi * w_expanded
    print("zw_roi['tas'].dims:", zw_roi['tas'].dims)
    print("zw_roi['tas'].shape:", zw_roi['tas'].shape)
    print(f"- zw_roi computed. Sample:\n{zw_roi['tas'].isel(time=0).sel(lat=45, lon=-105, method='nearest').values}")

    # Valid mask
    valid_mask = ~zw_roi['tas'].isnull()
    n = valid_mask.sum(dim=['lat', 'lon', 'time'])
    print(f"- valid_mask sum shape: {n.shape}, sample: {n.values[:5]}")

    # Aggregated scores
    abs_mean = zw_roi.apply(np.fabs).sum(dim=['lat', 'lon', 'time']) / n
    pos_mask = (valid_mask & (zw_roi > 0))
    neg_mask = (valid_mask & (zw_roi < 0))
    pos_mean = zw_roi.where(zw_roi > 0).sum(dim=['lat', 'lon', 'time']) / pos_mask.sum(dim=['lat', 'lon', 'time'])
    neg_mean = zw_roi.where(zw_roi < 0).sum(dim=['lat', 'lon', 'time']) / neg_mask.sum(dim=['lat', 'lon', 'time'])

    print(f"- abs_mean sample:\n{abs_mean['tas'].values[:5]}")
    print(f"- pos_mean sample:\n{pos_mean['tas'].values[:5]}")
    print(f"- neg_mean sample:\n{neg_mean['tas'].values[:5]}")

    scores = xr.Dataset(
        data_vars=dict(
            abs_mean=abs_mean['tas'].rename('abs_mean'),
            pos_mean=pos_mean['tas'].rename('pos_mean'),
            neg_mean=neg_mean['tas'].rename('neg_mean'),
        )
    )

    print(">>> Done.\n")
    return scores

def score_archetypes_in_roi(composite_da: xr.DataArray, roi_mask_da: xr.DataArray = None) -> xr.Dataset:
    """
    Compute abs_mean, pos_mean, neg_mean for each archetype 
    directly from composite snapshots.

    Parameters
    ----------
    composite_da : xr.DataArray
        3D array with dims ('archetype', 'lat', 'lon').
        Each entry is a composite snapshot for one archetype.
    roi_mask_da : xr.DataArray, optional
        2D mask (lat, lon). If None, the whole domain is used.

    Returns
    -------
    xr.Dataset with data_vars:
        abs_mean, pos_mean, neg_mean (dims: archetype)
    """
    
    # Default mask = whole field
    if roi_mask_da is None:
        roi_mask_da = xr.ones_like(composite_da.isel(archetype=0))
    
    # Apply mask
    comp_roi = composite_da.where(roi_mask_da == 1)

    # Absolute, positive, and negative mean anomalies
    abs_mean = comp_roi.pipe(np.abs).mean(dim=['lat', 'lon'])
    pos_mean = comp_roi.where(comp_roi > 0).mean(dim=['lat', 'lon'])
    neg_mean = comp_roi.where(comp_roi < 0).mean(dim=['lat', 'lon'])

    scores = xr.Dataset(
        data_vars=dict(
            abs_mean=abs_mean.rename('abs_mean'),
            pos_mean=pos_mean.rename('pos_mean'),
            neg_mean=neg_mean.rename('neg_mean'),
        )
    )

    return scores

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

    fname = os.path.join(plots_dir, f"composite_{n_arch}_arch_0d_{method}.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_composites_zscore_bbox(composite_da, plots_dir, method, varname='tas', zscore_bboxes = None, z_scores = None):
    # zscore_bboxes
    # np ndarray, holds [lat1, lat2, lon1, lon2], [lat1, lat2, lon1, lon2]
    # zscores
    # holds [pos_mean, neg_mean, abs_mean]
    # grouped by archetype number
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
    
        if zscore_bboxes is not None:
            for bbox in zscore_bboxes:
                lat1, lat2, lon1, lon2 = bbox
                width = lon2 - lon1
                height = lat2 - lat1

                rect = Rectangle(
                    (lon1, lat1), width, height,
                    linewidth=1, edgecolor='black', facecolor='none', hatch = '//',
                    transform=ccrs.PlateCarree()
                )
                ax.add_patch(rect)

        # Add global z-score annotations in bottom-right corner of figure
        if z_scores is not None and zscore_bboxes is not None:
            # Last entry in z_scores is assumed to be the global one
            label_texts = [
                f"Mean |z|: {z_scores['abs_mean'].sel(archetype = i).item():.3f}",
                f"Mean z⁺: {z_scores['pos_mean'].sel(archetype = i).item():.3f}",
                f"Mean z⁻: {z_scores['neg_mean'].sel(archetype = i).item():.3f}"
            ]

            lon, lat = 179, 28  # bottom right of global map
            for j, label in enumerate(label_texts):
                ax.text(
                    lon, lat - j*6, label,
                    ha='right', va='bottom',
                    fontsize=9,
                    transform=ccrs.PlateCarree()
                )

    fname = os.path.join(plots_dir, f"composite_{n_arch}_arch_0d_{method}_z.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main(method):
    repo_dir = "concurrent-heatwave-prediction"
    data_dir = os.path.join(repo_dir, "data", "lat30-60")
    plots_dir = os.path.join(repo_dir, "plots", "lat30-60")

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    tas_path = os.path.join(data_dir, 'lentis_tas_JJA_2deg_101_deseason.nc')
    pcha_path = os.path.join(data_dir, 'pcha_results_8a_0d.hdf5')

    dataset_tas = xr.open_dataset(tas_path)

    with h5py.File(pcha_path, 'r') as f:
        S_PCHA = f['/S_PCHA'][:]

    composite_da = compute_composites(dataset_tas, S_PCHA, method=method)
    # define the two regions (A for US, B for Europe)
    lat_a1, lat_a2, lon_a1, lon_a2 = 40, 55, -110, -75
    lat_b1, lat_b2, lon_b1, lon_b2 = 40, 55, -10, 45
    regions = np.array([[lat_a1, lat_a2, lon_a1, lon_a2], [lat_b1, lat_b2, lon_b1, lon_b2]])

    mask_US = build_box_mask_da(lat_a1, lat_a2, lon_a1, lon_a2, like_dataset=dataset_tas['tas'])
    assert(mask_US.sel(lat=39, lon=-111, method='nearest').item() == 0) # expect 0
    assert(mask_US.sel(lat=45, lon=-111, method='nearest').item() == 0) # expect 0
    assert(mask_US.sel(lat=56, lon=-74, method='nearest').item() == 0) # expect 0
    assert(mask_US.sel(lat=56, lon=-100, method='nearest').item() == 0) # expect 0
    assert(mask_US.sel(lat=41, lon=-109, method='nearest').item() == 1) # expect 1
    assert(mask_US.sel(lat=54, lon=-109, method='nearest').item() == 1) # expect 1
    assert(mask_US.sel(lat=45, lon=-100, method='nearest').item() == 1) # expect 1
    mask_EU = build_box_mask_da(lat_b1, lat_b2, lon_b1, lon_b2, like_dataset=dataset_tas['tas'])
    roi_mask_da = combine_masks(mask_EU, mask_US)
    print(roi_mask_da.sel(lat=41, lon=-109, method='nearest').item())
    print(roi_mask_da.dims)       # Should be ('lat', 'lon')
    print(roi_mask_da.coords)     # Should match z['tas'].isel(time=0).coords

    z_scores_da = score_archetypes_in_roi(composite_da, roi_mask_da)
    
    for arch in z_scores_da.archetype.values:
        print(f"Archetype: {arch}")
        for metric in z_scores_da.data_vars:
            val = z_scores_da[metric].sel(archetype=arch).item()
            print(f"  {metric}: {val:.3f}")
    
    plot_composites_zscore_bbox(composite_da, plots_dir, method, varname='tas', zscore_bboxes=regions, z_scores=z_scores_da)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and visualize TAS composite analysis conditioned on stream250 archetypes.')
    parser.add_argument('--method', choices=['argmax', 'weighted'],
                        help='Composite analysis method: "argmax" for hard grouping, "weighted" for soft composite')
    args = parser.parse_args()

    main(args.method)