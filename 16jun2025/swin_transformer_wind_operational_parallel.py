import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
import wandb
from swinir import SwinIR

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# torch.cuda.set_per_process_memory_fraction(0.9, device)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ----- Setup wandb -----
wandb.login(key="70f85253c59220a4439123cc3c97280ece560bf5")

# ----- DDP Setup -----
def setup_ddp():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

# ----- Dataset -----
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



#def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, epoch=None, best_model_path="best_model.pth"):
def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, epoch=None, best_val_loss=None, best_model_path="best_model.pth"):

    # global best_val_loss
    model.train()
    train_loss = 0.0

    for lr, hr in tqdm(train_dataloader, desc=f"Training Epoch {epoch}", leave=False):
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if epoch is not None:
            wandb.log({"batch_train_loss": loss.item(), "epoch": epoch})

    train_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr, hr in tqdm(val_dataloader, desc=f"Validation Epoch {epoch}", leave=False):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            val_loss += loss.item()
    # val_loss /= len(val_dataloader)
    val_loss_tensor = torch.tensor(val_loss, device=device)
    count_tensor = torch.tensor(len(val_dataloader), device=device)

    # Sum val_loss and sample count across all processes
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

    val_loss = val_loss_tensor.item() / count_tensor.item()


    if best_val_loss is None or val_loss < best_val_loss:
        if dist.get_rank() == 0:
            torch.save(model.module.state_dict(), best_model_path)
            print(f"✅ Saved new best model with val_loss: {val_loss:.4f}")
        best_val_loss = val_loss
        
    # if best_val_loss is None or val_loss < best_val_loss:
    #     if dist.get_rank() == 0:
    #         checkpoint = {
    #             "model_state_dict": model.module.state_dict(),
    #             "best_val_loss": val_loss
    #         }
    #         torch.save(checkpoint, best_model_path)
    #         print(f"✅ Saved new best model with val_loss: {val_loss:.4f}")
    # best_val_loss = val_loss


    torch.cuda.empty_cache()
    # return train_loss, val_loss
    return train_loss, val_loss, best_val_loss


# ----- Main daily training -----
def process_and_train_for_day(date_str, model, optimizer, criterion, device, local_rank, patch_size=256):
    log_path = "progress.log"
    if local_rank == 0 and os.path.exists(log_path):
        with open(log_path, "r") as f:
            if date_str in {line.strip() for line in f}:
                print(f"✅ {date_str} already completed.")
                return

    try:
        era5_path = f'/scratch/08105/ms86336/download_wind/usa_wind_{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}.nc'
        aorc_path = f'/scratch/08105/ms86336/download_noaa_aorc/noaa_aorc_usa/noaa_aorc_usa_{date_str[:4]}_day_{date_str}.nc'
        ds_era5 = xr.open_dataset(era5_path)
        ds_aorc = xr.open_dataset(aorc_path)
        ds_aorc = ds_aorc.assign_coords(longitude=ds_aorc.longitude + 360.0)

        ds_aorc = ds_aorc.sel(latitude=slice(25,49), longitude=slice(235,293))
        ds_era5 = ds_era5.sel(latitude=slice(25,49), longitude=slice(235,293))
        lats, lons = ds_aorc.latitude.values, ds_aorc.longitude.values
        ds_era5 = ds_era5.interp(latitude=lats, longitude=lons)

        u10_era5 = (ds_era5.u10.values + 32.257) / (31.689 + 32.257)
        v10_era5 = (ds_era5.v10.values + 35.598) / (36.181 + 35.598)
        u10_aorc = (ds_aorc.UGRD_10maboveground.values + 27.633) / (28.433 + 27.633)
        v10_aorc = (ds_aorc.VGRD_10maboveground.values + 29.667) / (34.417 + 29.667)

        x = np.stack([u10_era5, v10_era5], axis=1).astype(np.float32)
        y = np.stack([u10_aorc, v10_aorc], axis=1).astype(np.float32)
        x, y = x[:, :, :2816, :6912], y[:, :, :2816, :6912]
        x_patches = patchify_4d(x, patch_size)
        y_patches = patchify_4d(y, patch_size)
        x_patches = np.transpose(x_patches, (0, 2, 1, 3, 4)).reshape(-1, 2, patch_size, patch_size)
        y_patches = np.transpose(y_patches, (0, 2, 1, 3, 4)).reshape(-1, 2, patch_size, patch_size)

        mask = ~np.isnan(y_patches).any(axis=(1, 2, 3))
        x_clean, y_clean = x_patches[mask], y_patches[mask]

        split = int(0.8 * len(x_clean))
        train_dataset = ncDataset(x_clean[:split], y_clean[:split])
        val_dataset = ncDataset(x_clean[split:], y_clean[split:])

        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=2, pin_memory=True)
        
        # train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=0)
        # val_loader = DataLoader(val_dataset, batch_size=1, sampler=val_sampler, num_workers=0)
        # if local_rank == 0:
        #     wandb.init(project="wind_1km_1hr", name=f"swinir_{date_str}_rank{local_rank}", reinit=True)
        # ---- Load best_val_loss from checkpoint if available ----
        # best_val_loss = None
        # if os.path.exists("best_model.pth"):
        #     checkpoint = torch.load("best_model.pth", map_location=torch.device(f"cuda:{local_rank}"))
        #     model.module.load_state_dict(checkpoint["model_state_dict"])
        #     best_val_loss = checkpoint.get("best_val_loss", None)
        #     print(f"✅ Rank {local_rank} loaded best model with val_loss: {best_val_loss}")
        # checkpoint = torch.load("best_model.pth", map_location=torch.device(f"cuda:{local_rank}"))
        # model.module.load_state_dict(checkpoint["model_state_dict"])
        # best_val_loss = checkpoint.get("best_val_loss", None)
        # if os.path.exists("best_model.pth"):
        #     state_dict = torch.load("best_model.pth", map_location=torch.device(f"cuda:{local_rank}"))
        #     model.module.load_state_dict(state_dict)
        #     best_val_loss = None
        #     print(f"✅ Rank {local_rank} loaded best model.")
        #     # print(f"✅ Rank {local_rank} (global rank {dist.get_rank()}) loaded best model.")

        best_val_loss = None
        for epoch in range(1, 11):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            # train_loss, val_loss = train(model, train_loader, val_loader, criterion, optimizer, device, epoch)
            train_loss, val_loss, best_val_loss = train(
                model, train_loader, val_loader,
                criterion, optimizer, device,
                epoch=epoch,
                best_val_loss=best_val_loss
            )
            # if local_rank == 0:
            #     wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "date_str": date_str
            })
        # Load best model once on rank 0 and broadcast to others
        if dist.get_rank() == 0 and os.path.exists("best_model.pth"):
            state_dict = torch.load("best_model.pth", map_location=torch.device(f"cuda:{local_rank}"))
            model.module.load_state_dict(state_dict)
            print(f"✅ Rank {local_rank} loaded best model weights.")

        # Broadcast model parameters from rank 0 to all other ranks
        for param in model.module.parameters():
            dist.broadcast(param.data, src=0)

        # Load best model after training
        if os.path.exists("best_model.pth"):
            map_location = torch.device(f"cuda:{local_rank}")
            state_dict = torch.load("best_model.pth", map_location=map_location)
            model.module.load_state_dict(state_dict)
            print(f"✅ Rank {local_rank} loaded best model weights.")
            
        if local_rank == 0:
            with open(log_path, "a") as f:
                f.write(f"{date_str}\n")
        print(f"✅ Completed {date_str}")

    except Exception as e:
        print(f"❌ Error on {date_str}: {e}")
        if local_rank == 0:
            with open("error.log", "a") as f:
                f.write(f"{date_str}: {e}\n")

# ----- Main Entry -----
def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_per_process_memory_fraction(0.9, device)
    # ----- Training loop -----
    best_val_loss = None
    # Initialize wandb ONCE per rank
    wandb.init(
        project="wind_1km_1hr",
        name=f"swinir_ddp_shared_run_rank{local_rank}",  # Or no rank suffix if you only log from rank 0
        group="swinir_training_group",
        mode="online" if local_rank == 0 else "disabled"  # Enable only on rank 0
    )

    model = SwinIR(
        img_size=256,
        in_chans=2,
        embed_dim=60,
        depths=[6]*4,
        num_heads=[6]*4,
        window_size=8,
        mlp_ratio=2,
        upsampler='pixelshuffledirect'
    ).to(device)

    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # ----- Load best model if it exists -----
    # best_model_path = "best_model.pth"
    # if os.path.exists(best_model_path):
    #     map_location = torch.device(f"cuda:{local_rank}")
    #     state_dict = torch.load(best_model_path, map_location=map_location)
    #     model.module.load_state_dict(state_dict)
    #     print(f"✅ Rank {local_rank} loaded existing best model weights from '{best_model_path}'")


    start = datetime.strptime("20190101", "%Y%m%d")
    end = datetime.strptime("20211231", "%Y%m%d")
    current = start

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        process_and_train_for_day(date_str, model, optimizer, criterion, device, local_rank)
        current += timedelta(days=1)

    cleanup_ddp()

if __name__ == "__main__":
    main()
