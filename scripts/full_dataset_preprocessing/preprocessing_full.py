import argparse
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

def sqrt_cos_weighting(ds):
    weights = np.sqrt(np.cos(np.deg2rad(ds['lat'])))
    
    weighted = ds.copy()
    weighted['stream'] = ds['stream'] * weights
    
    return weighted

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
    print(f"\t{var} range: [{ds[var].min().values},{ds[var].max().values}], avg: {ds[var].mean().values}\n")

def preprocess_stream_1(ds):
    ds = pick_summer_months(ds)
    ds = downsample_spatial_resolution(ds)
    ds = slice_region(ds) 
    return ds

def preprocess_stream_2(ds):
    ds = deseasonalize(ds)
    ds = subtract_spatial_mean(ds)
    ds = sqrt_cos_weighting(ds)
    return ds

def main(args):
    if (args.stream):
        output_filename = "lentis_stream250_JJA_2deg_all_deseason_smsub_sqrtcosw.nc"
        filename_incl_path = "/scistor/ivm/data_catalogue/climate_models/lentis/day/stream250/stream_250_h_day_NH_ensemble1*.nc"

        ds_full = xr.open_mfdataset(filename_incl_path, combine = 'nested', concat_dim= 'time')
        """
        parts = []
        for i in range(1,17):
            print(f"Processing part {i} of 16...")
            double_digit_idx = str(i).zfill(2)
            filename_incl_path = f"/scistor/ivm/data_catalogue/climate_models/lentis/day/stream250/stream_250_h_day_NH_ensemble1{double_digit_idx}.nc"
            
            ds = xr.open_dataset(filename_incl_path)
            ds = ds.assign(part_idx=('time', [i] * ds.dims['time']))
            parts.append(ds)

            print(f"Done.")
        """
            
        ds_final = preprocess_stream_1(ds_full)
        print("=== Original shortened, coarsened, & sliced dataset: ===")
        sanity_check(ds_final, 'stream')
        ds_final = preprocess_stream_2(ds_final)

        print("=== Fully processed dataset: ===")
        sanity_check(ds_final, 'stream')
        
        ds_final.to_netcdf(output_filename)
        print(f"Dataset extract saved to {output_filename}")
    
    if (args.olr):
        print("Hello world!")



if __name__ == "__main__":
    print("Python script started")
    parser = argparse.ArgumentParser(description = 'Preprocessing operations for the full LENTIS dataset')
    parser.add_argument('--stream', action='store_true',
                        help="Preprocess stream function 250hPa variable")
    parser.add_argument('--tas', action='store_true',
                        help="Preprocess tas variable")
    parser.add_argument('--olr', action='store_true',
                        help="Preprocess olr variable")
    args = parser.parse_args()

    if not any([args.stream, args.tas, args.olr]):
        print("No preprocessing option selected. Use --stream, --tas, or --olr.")
    
    main(args)