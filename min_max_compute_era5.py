import xarray as xr

ds = xr.open_mfdataset('usa_*.nc')
print(ds.u10.max().compute())
print(ds.v10.max().compute())
print(ds.u10.min().compute())
print(ds.v10.min().compute())
