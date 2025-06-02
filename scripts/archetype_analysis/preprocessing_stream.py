import xarray as xr
import numpy as np

def pick_summer_months(ds):
    # if time column isn't in datetime format do:
    # ds['time'] = xr.decode_cf(ds).time
    summer_ds = ds.sel(time=ds['time'].dt.month.isin([6, 7, 8]))
    return summer_ds

def downsample_spatial_resolution(ds):
    # note: whether or not this indexing works depends on the main variable
    lat_step = float(ds['lat'][1] - ds['lat'][0])
    lon_step = float(ds['lon'][1] - ds['lon'][0])

    lat_factor = int(round(2 / lat_step))
    lon_factor = int(round(2 / lon_step))

    print(f"Original resolution: {lat_step}°, {lon_step}°")
    print(f"Coarsening factors: lat={lat_factor}, lon={lon_factor}\n")

    ds_coarse = ds.coarsen(lat = lat_factor, lon = lon_factor, boundary = 'trim').mean()
    return ds_coarse

def slice_region(ds):
    latmin, latmax, lonmin, lonmax = (15, 75, -180, 180)

    if np.diff(ds.lat)[0] < 0:
        # Spatial cropping, latitude is stored decreasing, longitude increasing
        subset = ds.sel(lat = slice(latmax, latmin), lon = slice(lonmin, lonmax))
    else: # latitude is increasing
        subset = ds.sel(lat = slice(latmin, latmax), lon = slice(lonmin, lonmax))

    return subset

# For each day, calculate the spatial mean
# and subtract it from every gridpoint
def subtract_spatial_mean(ds):
    #spatial_mean = ds.groupby('time').mean()
    #mean_subtracted = ds.groupby('time') - spatial_mean
    spatial_mean = ds['stream'].mean(dim=('lat', 'lon'))
    print(f"Spatial mean shape: {spatial_mean.shape}")

    mean_subtracted = ds.copy() # in order to maintain the dataset structure
    mean_subtracted['stream'] = ds['stream'] - spatial_mean

    return mean_subtracted

def deseasonalize(ds):
    # Group by day of year and calculate the daily climatology (mean for each calendar day)
    clim = ds.groupby('time.dayofyear').mean('time')

    # Subtract daily climatology from the original dataset
    deseasonalized = ds.groupby('time.dayofyear') - clim

    return deseasonalized

def sqrt_cos_weighting(ds):
    weights = np.sqrt(np.cos(np.deg2rad(ds['lat'])))
    
    weighted = ds.copy()
    weighted['stream'] = ds['stream'] * weights
    
    return weighted

def test_weighting(ds):
    # Indices for three test points
    # a and b at lat = 25 (should have same weight)
    # c at lat = 70 (should have different weight)
    
    # Find index where lat is close to 25
    lat_idx_25 = int(np.argmin(np.abs(ds['lat'].values - 25)))
    lat_idx_70 = int(np.argmin(np.abs(ds['lat'].values - 70)))

    # Pick arbitrary lon indices
    lon_idx_a = 10
    lon_idx_b = 30
    lon_idx_c = 50

    # Apply weighting
    ds_weighted = sqrt_cos_weighting(ds)

    # Access original and weighted values
    orig = ds['stream']
    weighted = ds_weighted['stream']
    
    t, p = 0, 0  # First time and pressure level

    val_a_orig = orig[t, p, lat_idx_25, lon_idx_a].item()
    val_b_orig = orig[t, p, lat_idx_25, lon_idx_b].item()
    val_c_orig = orig[t, p, lat_idx_70, lon_idx_c].item()

    val_a_wt = weighted[t, p, lat_idx_25, lon_idx_a].item()
    val_b_wt = weighted[t, p, lat_idx_25, lon_idx_b].item()
    val_c_wt = weighted[t, p, lat_idx_70, lon_idx_c].item()

    # Compute weights directly
    w_25 = np.sqrt(np.cos(np.deg2rad(ds['lat'].values[lat_idx_25])))
    w_70 = np.sqrt(np.cos(np.deg2rad(ds['lat'].values[lat_idx_70])))

    # Assertions
    assert np.allclose(val_a_wt, val_a_orig * w_25), "Point A incorrect"
    assert np.allclose(val_b_wt, val_b_orig * w_25), "Point B incorrect"
    assert np.allclose(val_c_wt, val_c_orig * w_70), "Point C incorrect"

    # Check that A and B got same weight
    rel_ratio_a = val_a_wt / val_a_orig
    rel_ratio_b = val_b_wt / val_b_orig
    assert np.allclose(rel_ratio_a, rel_ratio_b), "Points A and B weighted differently"

    # Check that A and C got different weights
    assert not np.allclose(w_25, w_70), "Weights for lat=25 and lat=70 are equal (they shouldn't be)"

    print("All weighting tests passed ✅")

# Example call (assuming you have `ds` defined):
# test_weighting(ds)

def sanity_check(ds, var):
    print("Dataset info")
    print(" Metadata")
    print(f"\tDimension names: {ds.dims}")
    print(f"\tAxis nums: {'lat'}->{ds[var].get_axis_num('lat')}, {'lon'}->{ds[var].get_axis_num('lon')}")
    print(f"\t{ds.coords}")
    
    print(" Shapes and numbers")
    print(f"\tDataset shape: {ds[var].data.shape}")
    print(f"\tNr. latitude entries: {ds['lat'].data.shape}, nr. longitude entries: {ds['lon'].data.shape}")
    print(f"\tLatitude range: [{ds['lat'].min().values},{ds['lat'].max().values}], longitude range: [{ds['lon'].min().values},{ds['lon'].max().values}]")
    print(f"\tSpatial resolution: {float(ds['lat'][1] - ds['lat'][0])}ºlat x {float(ds['lon'][1] - ds['lon'][0])}ºlon")
    print(f"\tNr. timesteps: {ds['time'].shape}")
    print(f"\tStream function range: [{ds[var].min().values},{ds[var].max().values}], avg: {ds[var].mean().values}\n")

def main():
    filename_incl_path = "/scistor/ivm/data_catalogue/climate_models/lentis/day/stream250/stream_250_h_day_NH_ensemble101.nc"
    output_filename = "lentis_stream250_JJA_2deg_101_deseason_smsubd_sqrtcosw.nc"
    
    dataset = xr.open_dataset(filename_incl_path)
    print("=== Original dataset: ===")
    sanity_check(dataset, 'stream')
    test_weighting(dataset)
    """
    dataset = pick_summer_months(dataset)
    dataset = downsample_spatial_resolution(dataset)
    dataset = slice_region(dataset)
    dataset = deseasonalize(dataset)
    dataset = subtract_spatial_mean(dataset)

    """
    print("=== Processed dataset: ===")
    sanity_check(dataset, 'stream')

    #dataset.to_netcdf(output_filename)
    print(f"Dataset extract saved to {output_filename}")

if __name__ == "__main__":
    main()