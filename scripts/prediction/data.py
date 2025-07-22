from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
import xarray as xr
import h5py
import numpy as np
from math import floor

# TODO change this to have all the final variables
def load_datasets(stream_path, tas_path):
    dataset_stream = xr.open_dataset(stream_path)
    dataset_tas = xr.open_dataset(tas_path)
    dataset_comb = dataset_stream.assign(tas=dataset_tas['tas'])
    return dataset_comb

def construct_targets_and_interpolate(dataset_comb, S_PCHA_path, lead_time, input_len = 1):
    with h5py.File(S_PCHA_path, 'r') as f:
        S_PCHA = f['/S_PCHA'][:]

    arch_indices = np.argmax(S_PCHA, axis=0)
    arch_da = xr.DataArray(arch_indices, dims="time", coords={"time": dataset_comb.time})
    arch_labels = arch_da.values

    stream = dataset_comb['stream'].squeeze('plev').values
    tas = dataset_comb['tas'].values
    x_np = np.stack([stream, tas], axis=-1)
    x_tensor = torch.from_numpy(x_np).float()

    x_tensor = interpolate_tensor(x_tensor)

    time = dataset_comb['time'].values
    x_list, y_list, kept_time_indices = [], [], []

    for t in range(len(time) - lead_time):
        target_time = time[t] + np.timedelta64(lead_time, 'D')
        if time[t + lead_time] == target_time:
            this_x_list = []
            for i in range (input_len):
              if i == 0:
                  this_x_list.append(x_tensor[t])
              else:
                  x_prev_time = time[t] - i*np.timedelta64(lead_time, 'D')
                  if (t - i >= 0) and time[t - i] == x_prev_time:
                      this_x_list.append(x_tensor[t-i])
                  else:
                      this_x_list.append(torch.zeros_like(x_tensor[t]))
            x_list.append(torch.stack(this_x_list)) # (T, H, W, C)
            y_list.append(arch_labels[t + lead_time])
            kept_time_indices.append(t)

    x_final = torch.stack(x_list)                       # shape: (N, T, H, W, C)
    y_final = torch.tensor(y_list, dtype=torch.long)

    return x_final, y_final, kept_time_indices


def interpolate_tensor(x_tensor, target_size=(128, 128)):
    x_tensor_perm = x_tensor.permute(0, 3, 1, 2)
    x_tensor_resized = F.interpolate(x_tensor_perm, size=target_size, mode='bilinear', align_corners=False)
    x_tensor_resized = x_tensor_resized.permute(0, 2, 3, 1)
    return x_tensor_resized

def split_data(x, y, split_ratio=0.8):
    split_idx = floor(x.shape[0] * split_ratio)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    return x_train, y_train, x_val, y_val

def get_dataloaders(x_train, y_train, x_val=None, y_val=None, batch_size=64):
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if x_val is not None and y_val is not None:
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
