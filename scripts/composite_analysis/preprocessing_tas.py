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

def deseasonalize(ds):
    # Group by day of year and calculate the daily climatology (mean for each calendar day)
    clim = ds.groupby('time.dayofyear').mean('time')

    # Subtract daily climatology from the original dataset
    deseasonalized = ds.groupby('time.dayofyear') - clim

    return deseasonalized

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
    print(f"\tTAS range: [{ds[var].min().values},{ds[var].max().values}], avg: {ds[var].mean().values}\n")

def main():
    filename_incl_path = "/scistor/ivm/data_catalogue/climate_models/lentis/day/tas/tas_h_day_NH_ensemble101.nc"
    output_filename = "lentis_tas_JJA_2deg_101_deseason.nc"
    
    dataset = xr.open_dataset(filename_incl_path)
    print("=== Original dataset: ===")
    sanity_check(dataset, 'tas')
    
    dataset = pick_summer_months(dataset)
    dataset = downsample_spatial_resolution(dataset)
    dataset = slice_region(dataset)
    dataset = deseasonalize(dataset)

    print("=== Processed dataset: ===")
    sanity_check(dataset, 'tas')

    dataset.to_netcdf(output_filename)
    print(f"Dataset extract saved to {output_filename}")

if __name__ == "__main__":
    main()