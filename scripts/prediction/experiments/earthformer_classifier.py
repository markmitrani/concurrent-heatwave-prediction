import os
import json
import h5py
import torch
import xarray as xr
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.utils.utils import download
from math import floor
from torch.utils.data import TensorDataset, DataLoader


def load_and_preprocess_data(stream_path, tas_path, pcha_path):
    dataset_stream = xr.open_dataset(stream_path)
    dataset_tas = xr.open_dataset(tas_path)

    with h5py.File(pcha_path, 'r') as f:
        S_PCHA = f['/S_PCHA'][:]

    arch_indices = np.argmax(S_PCHA, axis=0)
    dataset_comb = dataset_stream.assign(tas=dataset_tas['tas'])
    arch_da = xr.DataArray(arch_indices, dims="time", coords={"time": dataset_comb.time})
    dataset_comb = dataset_comb.assign(archetype=arch_da)

    stream = dataset_comb['stream'].squeeze('plev').values
    tas = dataset_comb['tas'].values
    x_np = np.stack([stream, tas], axis=-1)
    x_tensor = torch.from_numpy(x_np).float().permute(0, 3, 1, 2)
    x_tensor_resized = F.interpolate(x_tensor, size=(128, 128), mode='bilinear', align_corners=False)
    x_tensor = x_tensor_resized.permute(0, 2, 3, 1)

    time = dataset_comb['time'].values
    arch_labels = arch_da.values

    x_list, y_list, kept_indices = [], [], []
    lead = 7
    for t in range(len(time) - lead):
        if time[t + lead] == time[t] + np.timedelta64(lead, 'D'):
            x_list.append(x_tensor[t])
            y_list.append(arch_labels[t + lead])
            kept_indices.append(t)

    x_final = torch.stack(x_list)
    y_final = torch.tensor(y_list, dtype=torch.long)

    return x_final, y_final


def train_val_split(x, y, split_ratio=0.8):
    split_idx = floor(len(x) * split_ratio)
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]


def build_earthformer_model(config, checkpoint_url, save_dir):
    checkpoint_path = os.path.join(save_dir, "earthformer_earthnet2021.pt")
    download(url=checkpoint_url, path=checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model = CuboidTransformerModel(input_shape=[1, 128, 128, 2], target_shape=[1, 128, 128, 2], **config)
    model_state = model.state_dict()

    compatible = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    return model


class EarthformerClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.model = base_model
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(self.model.target_shape[-1], num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.pool(x).squeeze()
        logits = self.classifier(x)
        return torch.softmax(logits, dim=1)


def train_model(model, train_loader, val_loader, optimizer, loss_fn, num_epochs=20):
    train_loss_hist, val_loss_hist, val_acc_hist = [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss_hist.append(epoch_loss / len(train_loader))

        val_loss, val_acc = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = loss_fn(logits, yb)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == yb).float().mean().item()
                val_loss += loss.item()
                val_acc += acc

        val_loss_hist.append(val_loss / len(val_loader))
        val_acc_hist.append(val_acc / len(val_loader))

        print(f"Epoch {epoch+1}: Train Loss={train_loss_hist[-1]:.4f}, Val Loss={val_loss_hist[-1]:.4f}, Val Acc={val_acc_hist[-1]:.4f}")

    return train_loss_hist, val_loss_hist


def save_artifacts(model, optimizer, train_hist, val_hist):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"outputs/{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(out_dir, "checkpoint.pt"))

    with open(os.path.join(out_dir, "train_loss.json"), "w") as f:
        json.dump(train_hist, f)
    with open(os.path.join(out_dir, "val_loss.json"), "w") as f:
        json.dump(val_hist, f)

    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

def main():
    x, y = load_and_preprocess_data(
        "../../data/deseason_smsub/lentis_stream250_JJA_2deg_101_deseason_spatialsub.nc",
        "../../data/deseason_smsub/lentis_tas_JJA_2deg_101_deseason.nc",
        "../../data/deseason_smsub/pcha_results_8a.hdf5"
    )

    x_train, x_val, y_train, y_val = train_val_split(x, y)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32)

    config = { ... }  # Your Earthformer config dict here
    EF_model = build_earthformer_model(config, "https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_earthnet2021.pt", "./experiments")

    clf = EarthformerClassifier(EF_model, num_classes=8)
    for param in EF_model.parameters():
        param.requires_grad = False
    for param in clf.classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, clf.parameters()), lr=5e-4)
    loss_func = nn.CrossEntropyLoss()

    train_hist, val_hist = train_model(clf, train_loader, val_loader, optimizer, loss_func)
    save_artifacts(clf, optimizer, train_hist, val_hist)

if __name__ == "__main__":
    main()
    