import fsspec
import xarray as xr

reanalysis = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr',
    chunks={'time': 48},storage_options={'token': 'anon'},
    consolidated=True
)
print(reanalysis)

import xarray as xr
import scipy.spatial
import numpy as np

def mirror_point_at_360(ds):
    # Compute the mask to avoid Dask shape ambiguity
    mask = (ds.longitude == 0).compute()
    extra_point = (
        ds.where(mask, drop=True)
        .assign_coords(longitude=lambda x: x.longitude + 360)
    )
    return xr.concat([ds, extra_point], dim='values')


def build_triangulation(x, y):
  grid = np.stack([x, y], axis=1)
  return scipy.spatial.Delaunay(grid)
def build_triangulation(x, y):
  grid = np.stack([x, y], axis=1)
  return scipy.spatial.Delaunay(grid)

def interpolate(data, tri, mesh):
  indices = tri.find_simplex(mesh)
  ndim = tri.transform.shape[-1]
  T_inv = tri.transform[indices, :ndim, :]
  r = tri.transform[indices, ndim, :]
  c = np.einsum('...ij,...j', T_inv, mesh - r)
  c = np.concatenate([c, 1 - c.sum(axis=-1, keepdims=True)], axis=-1)
  result = np.einsum('...i,...i', data[:, tri.simplices[indices]], c)
  return np.where(indices == -1, np.nan, result)

longitude = np.linspace(0, 360, num=360*4+1)
latitude = np.linspace(-90, 90, num=180*4+1)
mesh = np.stack(np.meshgrid(longitude, latitude, indexing='ij'), axis=-1)

u10 = reanalysis['u10']
v10 = reanalysis['v10']
def lon_to_360(dlon):
    return ((360 + (dlon % 360)) % 360)


lon_west = lon_to_360(-106.65)
lon_east = lon_to_360(-93.51)
lat_south = 25.84
lat_north = 36.50

# Compute the mask explicitly
mask = (
    (reanalysis.longitude >= lon_west) & (reanalysis.longitude <= lon_east) &
    (reanalysis.latitude >= lat_south) & (reanalysis.latitude <= lat_north)
).compute()

# Apply the mask to subset and drop unneeded values
texas_ds = reanalysis[['u10', 'v10']].where(mask, drop=True)


import numpy as np
import xarray as xr
import scipy.spatial
from tqdm import tqdm
import pandas as pd

# ----------- Helper Functions -----------
def lon_to_360(dlon):
    return ((360 + (dlon % 360)) % 360)

def interpolate_time_series(var, tri, mesh, lon_new, lat_new):
    time_len = var.shape[0]
    result = np.empty((time_len, len(lon_new), len(lat_new)))

    for t in range(time_len):
        values = var.isel(time=t).values.ravel()
        indices = tri.find_simplex(mesh)
        T_inv = tri.transform[indices, :2, :]
        r = tri.transform[indices, 2, :]
        c = np.einsum('...ij,...j', T_inv, mesh - r)
        c = np.concatenate([c, 1 - c.sum(axis=-1, keepdims=True)], axis=-1)
        interp_vals = np.einsum('...i,...i', values[tri.simplices[indices]], c)
        interp_vals[indices == -1] = np.nan
        result[t] = interp_vals.reshape(len(lon_new), len(lat_new))

    return result


# ----------- Define US Grid and Subset -----------
lon_west = lon_to_360(-125.0)
lon_east = lon_to_360(-66.5)
lat_south = 24.5
lat_north = 49.5

mask = (
    (reanalysis.longitude >= lon_west) & (reanalysis.longitude <= lon_east) &
    (reanalysis.latitude >= lat_south) & (reanalysis.latitude <= lat_north)
).compute()

texas_ds = reanalysis[['u10', 'v10']].where(mask, drop=True)

# Grid for interpolation
lon_new = np.arange(lon_west, lon_east + 0.25, 0.25)
lat_new = np.arange(lat_south, lat_north + 0.25, 0.25)
lon_grid, lat_grid = np.meshgrid(lon_new, lat_new, indexing='ij')
mesh = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=-1)

# Triangulation (once)
lon = texas_ds.longitude.values.ravel()
lat = texas_ds.latitude.values.ravel()
tri = scipy.spatial.Delaunay(np.stack([lon, lat], axis=1))


# ----------- Loop over years and days -----------
for year in range(2018, 2025):
    print(f"\n🔄 Processing year: {year}")
    year_ds = texas_ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))

    if year_ds.time.size == 0:
        print(f"⚠️ No data found for {year}, skipping.")
        continue

    # Get unique days
    dates = pd.to_datetime(year_ds.time.values).normalize()
    unique_days = sorted(set(dates))

    for day in tqdm(unique_days, desc=f"📅 Saving daily files for {year}"):
        day_str = str(day.date())
        day_ds = year_ds.sel(time=day_str)

        if day_ds.time.size == 0:
            continue

        # Interpolate
        u10_interp = interpolate_time_series(day_ds.u10, tri, mesh, lon_new, lat_new)
        v10_interp = interpolate_time_series(day_ds.v10, tri, mesh, lon_new, lat_new)

        # Create DataArrays
        times = day_ds.time.values
        u10_da = xr.DataArray(u10_interp, coords=[('time', times), ('longitude', lon_new), ('latitude', lat_new)])
        v10_da = xr.DataArray(v10_interp, coords=[('time', times), ('longitude', lon_new), ('latitude', lat_new)])

        # Combine and save
        daily_wind = xr.Dataset({'u10': u10_da, 'v10': v10_da})
        daily_wind = daily_wind.transpose('time', 'latitude', 'longitude')
        encoding = {var: {"zlib": True, "complevel": 4} for var in daily_wind.data_vars}
        daily_wind.to_netcdf(f"texas_wind_{day_str}.nc", encoding=encoding)
