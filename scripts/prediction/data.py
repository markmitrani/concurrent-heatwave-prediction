from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
import xarray as xr
import h5py
import numpy as np
from math import floor

def load_datasets(stream_path, olr_path):
    dataset_stream = xr.open_dataset(stream_path)
    dataset_olr = xr.open_dataset(olr_path)
    return dataset_stream, dataset_olr

def extend_and_combine_datasets(stream_ds, olr_ds, olr_lag = 0):
    target_lats = stream_ds['lat']
    spacing = np.mean(np.diff(target_lats))

    start = target_lats[0] - spacing
    n_points = 17
    new_lats = np.linspace(start, start - spacing * (n_points-1), n_points)
    extended_target_lats = np.concatenate([new_lats[::-1], target_lats])
    aligned_olr_ds = olr_ds.interp(lat=extended_target_lats, method='linear')

    stream_ds_ext = stream_ds.interp(lat=extended_target_lats, method='linear')
    stream_np = stream_ds_ext['stream'].squeeze('plev').values

    # olr input lags behind stream func data by predefined amount of days
    aligned_olr_ds['time'] = aligned_olr_ds['time'] + np.timedelta64(olr_lag, 'D')

    aligned_olr_jja = aligned_olr_ds.sel(time=aligned_olr_ds['time'].dt.month.isin([6, 7, 8]))
    olr_np = aligned_olr_jja['rlut'].values

    x_np = np.stack([stream_np, olr_np], axis=-1)
    
    return x_np

# Modified target construction method for regression task
def construct_targets_and_interpolate(x_np, stream_ds, S_PCHA_path, lead_time, archetype_index=3, input_len=1, window_size=None):
    with h5py.File(S_PCHA_path, 'r') as f:
        S_PCHA = f['/S_PCHA'][:]

    print("S_PCHA shapes:")
    print(S_PCHA.shape)

    arch_probs = S_PCHA[archetype_index]  # shape: (time,)
    target_da = xr.DataArray(arch_probs, dims="time", coords={"time": stream_ds.time})
    targets = target_da.values               # numpy array of full-length targets

    x_tensor = torch.from_numpy(x_np).float()
    x_tensor = interpolate_tensor_with_padding(x_tensor)

    time = stream_ds['time'].values
    x_list, y_list, kept_time_indices = [], [], []

    for t in range(len(time) - lead_time):
        target_time = time[t] + np.timedelta64(lead_time, 'D')
        if time[t + lead_time] == target_time:
            this_x_list = []
            for i in range(input_len):
                if i == 0:
                    this_x_list.append(x_tensor[t])
                else:
                    x_prev_time = time[t] - i * np.timedelta64(lead_time, 'D')
                    if (t - i >= 0) and time[t - i] == x_prev_time:
                        this_x_list.append(x_tensor[t - i])
                    else:
                        this_x_list.append(torch.zeros_like(x_tensor[t]))
            x_list.append(torch.stack(this_x_list))  # (T, H, W, C)
            y_list.append(targets[t + lead_time])
            kept_time_indices.append(t)

    x_final = torch.stack(x_list)                       # shape: (N, T, H, W, C)

    # original selected targets and their corresponding target times
    y_final = torch.tensor(y_list, dtype=torch.float)  # shape (N,)
    kept_time_indices = np.array(kept_time_indices, dtype=int)

    print(f"x final shape: {x_final.shape}")
    print(f"y final shape: {y_final.shape}")
    print(f"y mean: {y_final.mean()}")
    print(f"baseline accuracy: {(np.abs(y_final - 0.15) < 0.05).sum() / y_final.shape[0]}")

    # apply rolling avg after selection
    if window_size is not None and window_size > 1 and y_final.numel() > 0:
        # compute the target times corresponding to each kept sample
        # note: the y value was targets[t + lead_time], so index into time is t + lead_time
        target_indices = kept_time_indices + lead_time
        target_times_selected = time[target_indices]

        # find breaks where consecutive selected target times are not consecutive days
        # we treat day-step as 1 day; convert diff to integer days
        time_deltas = np.diff(target_times_selected).astype('timedelta64[D]').astype(int)
        breaks = np.where(time_deltas != 1)[0] + 1  # break positions in the selected sequence

        # segment boundaries (start inclusive, end exclusive)
        segment_starts = np.concatenate(([0], breaks))
        segment_ends = np.concatenate((breaks, [len(y_final)]))

        print(segment_starts)

        y_vals = y_final.numpy().astype(float)  # work in numpy for cumsum
        y_smoothed = np.empty_like(y_vals)

        for start, end in zip(segment_starts, segment_ends):
            seg = y_vals[start:end]
            n = len(seg)
            if n == 0:
                continue

            # fast cumulative-sum based rolling:
            # sum[i] = sum_{0..i} seg[j]
            sums = np.cumsum(seg, dtype=float)

            # counts for each position: min(i+1, window_size)
            # sums for i < window_size: sums[i]
            # sums for i >= window_size: sums[i] - sums[i-window_size]
            out = np.empty_like(seg)

            # vectorized handling:
            if n <= window_size:
                # for all indices here, we average over available history (1..i+1)
                counts = np.arange(1, n + 1)
                out = sums / counts
            else:
                # first window_size entries averaged over fewer elements
                counts_head = np.arange(1, window_size + 1)
                out[:window_size] = sums[:window_size] / counts_head
                # remaining entries: sliding window of fixed size
                sums_tail = sums[window_size:] - sums[:-window_size]
                out[window_size:] = sums_tail / window_size

            y_smoothed[start:end] = out

        # replace y_final with smoothed values (convert back to torch tensor)
        y_final = torch.tensor(y_smoothed, dtype=torch.float)

        print(f"y final shape: {y_final.shape}")
        print(f"y mean: {y_final.mean()}")
        print(f"baseline accuracy: {(np.abs(y_final - 0.15) < 0.05).sum() / y_final.shape[0]}")

    return x_final, y_final, kept_time_indices

def interpolate_tensor_with_padding(x_tensor, target_size=(128, 128)):
    h,w = x_tensor.shape[1:3]
    print(f"w:{w}, h:{h}")
    target_w = target_size[1]
    scale = target_w/w
    target_h = int(round(h*scale))

    x_tensor_perm = x_tensor.permute(0, 3, 1, 2) # now T,C,H,W
    x_tensor_resized = F.interpolate(x_tensor_perm, size=(target_h, target_w), mode='bilinear', align_corners=False)
    x_tensor_resized = x_tensor_resized.permute(0, 2, 3, 1) # now T, H_new, W_new, C

    print(x_tensor_resized.shape)

    # Center the resized tensor along height
    out = torch.full((x_tensor.shape[0], target_size[0], target_size[1], x_tensor.shape[3]), float('nan'))
    top = (target_size[0] - target_h) // 2

    out[:, top:top + target_h, :target_w, :] = x_tensor_resized
    print(f"Out shape: {out.shape}")
    out_no_nan = torch.nan_to_num(out, nan=0.0)
    return out_no_nan

def split_data(x, y, split_ratio=0.8):
    split_idx = floor(x.shape[0] * split_ratio)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    return x_train, y_train, x_val, y_val

def minmax_scale(train_data: torch.Tensor, val_data: torch.Tensor):
    """
    Min-max scale train and val to [0,1] using stats from train_data.
    Works with data shaped (N, T, H, W, C).
    """
    min_vals = train_data.amin(dim=(0,1,2,3), keepdim=True)  # needs torch >=1.7
    max_vals = train_data.amax(dim=(0,1,2,3), keepdim=True)

    denom = torch.where(max_vals - min_vals == 0, torch.tensor(1.0, device=train_data.device), max_vals - min_vals)

    scaled_train = (train_data - min_vals) / denom
    scaled_val   = (val_data   - min_vals) / denom
    return scaled_train, scaled_val, min_vals, max_vals


def get_dataloaders(x_train, y_train, x_val, y_val, batch_size=64):
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

"""
def construct_targets_and_interpolate_stream_tas(dataset_comb, S_PCHA_path, lead_time, archetype_index=3, input_len = 1):
    with h5py.File(S_PCHA_path, 'r') as f:
        S_PCHA = f['/S_PCHA'][:]

    print("S_PCHA shapes:")
    print(S_PCHA.shape)
    # for i in range(S_PCHA.shape[0]):
    #     print(f"Arch {i} mean: {S_PCHA[i].mean()}")
    #     print(f"Arch {i} variance: {S_PCHA[i].var()}")

    arch_probs = S_PCHA[archetype_index]  # shape: (time,)
    target_da = xr.DataArray(arch_probs, dims="time", coords={"time": dataset_comb.time})
    targets = target_da.values

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
            y_list.append(targets[t + lead_time])
            kept_time_indices.append(t)

    x_final = torch.stack(x_list)                       # shape: (N, T, H, W, C)
    y_final = torch.tensor(y_list, dtype=torch.float)

    return x_final, y_final, kept_time_indices

def interpolate_tensor(x_tensor, target_size=(128, 128)):
    x_tensor_perm = x_tensor.permute(0, 3, 1, 2)
    x_tensor_resized = F.interpolate(x_tensor_perm, size=target_size, mode='bilinear', align_corners=False)
    x_tensor_resized = x_tensor_resized.permute(0, 2, 3, 1)
    return x_tensor_resized

"""

"""
def construct_targets_and_interpolate_old(dataset_comb, S_PCHA_path, lead_time, input_len = 1):
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
"""