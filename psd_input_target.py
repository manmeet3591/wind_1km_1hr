import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys

# Load datasets
ds_era5 = xr.open_dataset('/scratch/08105/ms86336/download_wind/usa_wind_2021-08-31.nc')
ds_aorc = xr.open_dataset('/scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/noaa_aorc_usa_2021_day_20210831.nc')

# Select the first time slice
u10_era5 = ds_era5.u10.isel(time=0)
v10_era5 = ds_era5.v10.isel(time=0)
u10_aorc = ds_aorc.UGRD_10maboveground.isel(time=0)
v10_aorc = ds_aorc.VGRD_10maboveground.isel(time=0)

# Extract timestamps
time_era5 = str(ds_era5.time.isel(time=0).values)
time_aorc = str(ds_aorc.time.isel(time=0).values)

# Set color range for map plots
vmin, vmax = -20, 20

# --- PSD Analysis ---

# Define US land bounding box
lat_box = slice(25, 50)
lon_box = slice(235, 295) if u10_aorc.longitude.max() > 180 else slice(-125, -65)

# Spatial subset for AORC
u10_aorc_sub = u10_aorc.sel(latitude=lat_box, longitude=lon_box)
v10_aorc_sub = v10_aorc.sel(latitude=lat_box, longitude=lon_box)

lon_box = slice(235, 295) if u10_era5.longitude.max() > 180 else slice(-125, -65)
# Spatial subset for ERA5
#print(u10_era5)
#print(lon_box)
#sys.exit()
u10_era5_sub = u10_era5.sel(latitude=lat_box, longitude=lon_box)
v10_era5_sub = v10_era5.sel(latitude=lat_box, longitude=lon_box)
#print(u10_era5_sub)
#sys.exit()
# PSD function
def compute_psd(da):
    if 'lat' in da.dims:
        da = da.rename({'lat': 'latitude'})
    if 'lon' in da.dims:
        da = da.rename({'lon': 'longitude'})
    da = da.where(~np.isnan(da), drop=True)
    n_lat = da.sizes['latitude']
    n_lon = da.sizes['longitude']
    da = da.fillna(0)
    dft = np.fft.fft(da, axis=1) / n_lon
    dft_amplitude = np.abs(dft)
    energy = 2 * (dft_amplitude ** 2)
    energy[..., 0] /= 2
    energy = energy[:, :n_lon // 2]
    energy_da = xr.DataArray(
        energy,
        dims=['latitude', 'wavenumber'],
        coords={'latitude': da['latitude'], 'wavenumber': np.arange(n_lon // 2)}
    )
    R = 6371e3
    circumference = 2 * np.pi * R * np.cos(np.radians(energy_da['latitude']))
    weighted_energy = energy_da * circumference
    psd = weighted_energy.mean(dim='latitude')
    return psd

# Compute PSDs
psd_u_aorc = compute_psd(u10_aorc_sub)
psd_v_aorc = compute_psd(v10_aorc_sub)
psd_u_era5 = compute_psd(u10_era5_sub)
psd_v_era5 = compute_psd(v10_era5_sub)

# Plot all PSDs
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# AORC u10
axs[0].loglog(psd_u_aorc['wavenumber'], psd_u_aorc, label='u10 NOAA AORC', linewidth=2, color='red')
axs[0].loglog(psd_u_era5['wavenumber'], psd_u_era5, label='u10 ERA5', linewidth=2, color='green')
axs[0].set_title('Zonal PSD of u10 (NOAA AORC) v/s u10 (ERA5)')
axs[0].set_xlabel('Wavenumber')
axs[0].set_ylabel('Power Spectral Density')
axs[0].legend()
axs[0].grid(True, which='both', linestyle='--', alpha=0.6)

# AORC v10
axs[1].loglog(psd_v_aorc['wavenumber'], psd_v_aorc, label='v10 AORC', linewidth=2, color='orange')
axs[1].loglog(psd_v_era5['wavenumber'], psd_v_era5, label='v10 ERA5', linewidth=2, color='black')
axs[1].set_title('Zonal PSD of v10 (NOAA AORC) v/s v10 (ERA5)')
axs[1].set_xlabel('Wavenumber')
axs[1].set_ylabel('Power Spectral Density')
axs[1].grid(True, which='both', linestyle='--', alpha=0.6)
axs[1].legend()
plt.tight_layout()
plt.show()
