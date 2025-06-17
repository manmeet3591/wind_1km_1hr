import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
from copy import deepcopy
# from swin_transformer_wind import SwinIR
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from swinir import SwinIR

# ----- Setup wandb -----
wandb.login(key="70f85253c59220a4439123cc3c97280ece560bf5")

# ----- Dataset class -----
class ncDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.targets[index])

    def __len__(self):
        return len(self.data)

# ----- Patch utils -----
def patchify_4d(data, patch_size):
    T, C, H, W = data.shape
    H_patches = H // patch_size
    W_patches = W // patch_size
    patches = data.reshape(T, C, H_patches, patch_size, W_patches, patch_size)
    patches = patches.transpose(0, 1, 2, 4, 3, 5).reshape(T, C, H_patches * W_patches, patch_size, patch_size)
    return patches

def unpatchify_4d(patches, original_shape):
    T, C, N, ph, pw = patches.shape
    H, W = original_shape[2], original_shape[3]
    H_patches = H // ph
    W_patches = W // pw
    data = patches.reshape(T, C, H_patches, W_patches, ph, pw)
    return data.transpose(0, 1, 2, 4, 3, 5).reshape(T, C, H, W)

# ----- Training function -----
#def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, epoch=None, best_model_path="best_model.pth"):
def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, epoch=None, best_val_loss=None, best_model_path="best_model.pth"):

    model.train()
    train_loss = 0.0
    for lr, hr in tqdm(train_dataloader, desc=f"Training Epoch {epoch}", leave=False):
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        train_loss += batch_loss
        if epoch is not None:
            wandb.log({"batch_train_loss": batch_loss, "epoch": epoch})
    train_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr, hr in tqdm(val_dataloader, desc=f"Validation Epoch {epoch}", leave=False):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            val_loss += loss.item()
    val_loss /= len(val_dataloader)
    # Checkpoint the best model
    if best_val_loss is None or val_loss < best_val_loss:
        torch.save(model.state_dict(), best_model_path)
        best_val_loss = val_loss
        print(f"✅ Saved new best model with val_loss: {val_loss:.4f}")
    return train_loss, val_loss, best_val_loss

# ----- Main daily processing function -----
def process_and_train_for_day(date_str, model, optimizer, criterion, device, patch_size=256):
    log_path = "progress.log"
    best_model_path = f"swinir_best_model.pth"
    best_val_loss = None
    # Optionally load previous best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("📦 Loaded previous best model weights.")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            if date_str in {line.strip() for line in f}:
                print(f"✅ {date_str} already completed.")
                return

    try:
        # Load datasets
        era5_path = f'/scratch/08105/ms86336/download_wind/usa_wind_{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}.nc'
        aorc_path = f'/scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/noaa_aorc_usa_{date_str[:4]}_day_{date_str}.nc'
        ds_era5 = xr.open_dataset(era5_path)
        ds_aorc = xr.open_dataset(aorc_path)
        # ds_aorc['longitude'] += 360.0
        ds_aorc = ds_aorc.assign_coords(longitude=ds_aorc.longitude + 360.0)

        ds_aorc = ds_aorc.sel(latitude=slice(25,49), longitude=slice(235,293))
        ds_era5 = ds_era5.sel(latitude=slice(25,49), longitude=slice(235,293))
        lats, lons = ds_aorc.latitude.values, ds_aorc.longitude.values
        ds_era5 = ds_era5.interp(latitude=lats, longitude=lons)

        u10_era5 = ds_era5.u10.values
        v10_era5 = ds_era5.v10.values
        u10_aorc = ds_aorc.UGRD_10maboveground.values
        v10_aorc = ds_aorc.VGRD_10maboveground.values

        # Min-max normalization
        u10_era5 = (u10_era5 + 32.257) / (31.689 + 32.257)
        v10_era5 = (v10_era5 + 35.598) / (36.181 + 35.598)
        u10_aorc = (u10_aorc + 27.633) / (28.433 + 27.633)
        v10_aorc = (v10_aorc + 29.667) / (34.417 + 29.667)

        # Prepare data
        x = np.stack([u10_era5, v10_era5], axis=1).astype(np.float32)
        y = np.stack([u10_aorc, v10_aorc], axis=1).astype(np.float32)
        x, y = x[:, :, :2816, :6912], y[:, :, :2816, :6912]
        x_patches = patchify_4d(x, patch_size)
        y_patches = patchify_4d(y, patch_size)
        x_patches = np.transpose(x_patches, (0, 2, 1, 3, 4)).reshape(-1, 2, patch_size, patch_size)
        y_patches = np.transpose(y_patches, (0, 2, 1, 3, 4)).reshape(-1, 2, patch_size, patch_size)

        # Remove NaNs
        mask = ~np.isnan(y_patches).any(axis=(1, 2, 3))
        x_clean, y_clean = x_patches[mask], y_patches[mask]

        # Train-validation split
        split = int(0.8 * len(x_clean))
        train_dataset = ncDataset(x_clean[:split], y_clean[:split])
        val_dataset = ncDataset(x_clean[split:], y_clean[split:])
        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

        wandb.init(project="wind_1km_1hr", name=f"swinir_{date_str}", reinit=True)
        for epoch in range(1, 11):  # Change 11 to more epochs if needed
#            train_loss, val_loss = train(model, train_loader, val_loader, criterion, optimizer, device, epoch, best_model_path=f"swinir_best_model.pth")
            train_loss, val_loss, best_val_loss = train(
                model, train_loader, val_loader, criterion, optimizer, device,
                epoch, best_val_loss, best_model_path
            )
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        with open(log_path, "a") as f:
            f.write(f"{date_str}\n")
        print(f"✅ Completed {date_str}")

    except Exception as e:
        print(f"❌ Error on {date_str}: {e}")
        with open("error.log", "a") as f:
            f.write(f"{date_str}: {e}\n")

# ----- Run over all dates -----
# from swinir import SwinIR  # You must import or define the SwinIR class beforehand

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SwinIR(img_size=256, in_chans=2, embed_dim=60, depths=[6]*4, num_heads=[6]*4,
               window_size=8, mlp_ratio=2, upsampler='pixelshuffledirect').to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

start = datetime.strptime("20190101", "%Y%m%d")
end = datetime.strptime("20211231", "%Y%m%d")
current = start

while current <= end:
    date_str = current.strftime("%Y%m%d")
    process_and_train_for_day(date_str, model, optimizer, criterion, device)
    current += timedelta(days=1)
