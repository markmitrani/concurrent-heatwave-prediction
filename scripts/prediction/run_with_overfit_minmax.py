from datetime import datetime
import math
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import data
import model
import utils
import matplotlib.pyplot as plt

# source: https://github.com/amazon-science/earth-forecasting-transformer/blob/main/scripts/cuboid_transformer/earthnet_w_meso/earthformer_earthnet_v1.yaml
earthformer_config = {
    "base_units": 256,
    "block_units": None,
    "scale_alpha": 1.0,

    "enc_depth": [1, 1],
    "dec_depth": [1, 1],
    "enc_use_inter_ffn": True,
    "dec_use_inter_ffn": True,
    "dec_hierarchical_pos_embed": False,

    "downsample": 2,
    "downsample_type": "patch_merge",
    "upsample_type": "upsample",

    "num_global_vectors": 2,
    "use_dec_self_global": True,
    "dec_self_update_global": True,
    "use_dec_cross_global": True,
    "use_global_vector_ffn": True,
    "use_global_self_attn": True,
    "separate_global_qkv": True,
    "global_dim_ratio": 1,

    "attn_drop": 0.0,
    "proj_drop": 0.0,
    "ffn_drop": 0.0,
    "num_heads": 4,

    "ffn_activation": "gelu",
    "gated_ffn": False,
    "norm_layer": "layer_norm",
    "padding_type": "zeros",
    "pos_embed_type": "t+hw",
    "use_relative_pos": True,
    "self_attn_use_final_proj": True,

    "checkpoint_level": 0,

    "initial_downsample_type": "stack_conv",
    "initial_downsample_activation": "leaky",
    "initial_downsample_stack_conv_num_layers": 2,
    "initial_downsample_stack_conv_dim_list": [64, 256],
    "initial_downsample_stack_conv_downscale_list": [2, 2],
    "initial_downsample_stack_conv_num_conv_list": [2, 2],

    "attn_linear_init_mode": "0",
    "ffn_linear_init_mode": "0",
    "conv_init_mode": "0",
    "down_up_linear_init_mode": "0",
    "norm_init_mode": "0",

    "padding_type": "ignore",
    "dec_cross_last_n_frames": None
}

def main():
    assert torch.cuda.is_available(), "CUDA is not available. Please run on a GPU-enabled machine."

    tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Experimental variables
    input_len = 5
    lead_time = 5
    batch_size = 16
    num_epochs = 100
    num_classes = 8
    archetype_index = 3

    # Optimizer parameters
    adam_lr = 5e-4
    adamw_lr = 1e-5
    betas = (0.9, 0.999)
    weight_decay = 1e-5

    # import the data
    stream_path = "data/lentis_stream.nc"
    olr_path = "data/lentis_olr.nc"
    S_PCHA_path = "data/pcha.hdf5"
    stream_ds, olr_ds = data.load_datasets(stream_path, olr_path)
    x_np = data.extend_and_combine_datasets(stream_ds, olr_ds)

    x, y, _ = data.construct_targets_and_interpolate(x_np, stream_ds, S_PCHA_path, lead_time, archetype_index, input_len = input_len)

    x_train, y_train, x_val, y_val = data.split_data(x,y)

    x_train, x_val, min_vals, max_vals = data.minmax_scale(x_train, x_val)

    train_loader, val_loader = data.get_dataloaders(x_train[:100], y_train[:100], x_train[:100], y_train[:100], batch_size)

    # train_loader=train_loader[:100]
    # val_loader=val_loader[:100]

    # URL to retrieve pretrained weights
    pretrained_checkpoint_url = "https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_earthnet2021.pt"
    save_dir = "pretrained/"

    EFCmodel, compatible = model.build_full_model(num_classes, input_len, earthformer_config, pretrained_checkpoint_url, save_dir)

    # test_model = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(x_train.shape[1] * x_train.shape[2] * x_train.shape[3] * x_train.shape[4], 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 1)
    #     )
    
    # EFCmodel = test_model

    EFCmodel.to("cuda")

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, EFCmodel.parameters()), lr=adam_lr)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, EFCmodel.parameters()),
                            lr=adamw_lr, betas=betas, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, utils.get_lr_lambda(num_epochs//5, num_epochs))
    criterion = nn.MSELoss()
    train_loss_history, val_loss_history, lr_history = [], [], []

    for epoch in range(num_epochs):
        EFCmodel.train()
        total_train_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as prog:
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            for batch_x, batch_y in prog:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")
                optimizer.zero_grad()
                logits = EFCmodel(batch_x)

                loss = criterion(logits.squeeze(), batch_y)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                prog.set_postfix(loss=loss.item())

            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        EFCmodel.eval()
        total_val_loss = 0.0
        # total_correct = 0
        # total_samples = 0

        # if epoch % 10 == 0:
        all_preds = []
        all_targets = []
        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to("cuda")
                batch_y = batch_y.to("cuda")

                logits = EFCmodel(batch_x)
                preds = logits.squeeze()

                loss = criterion(preds, batch_y)
                total_val_loss += loss.item()

                all_preds.append(logits.squeeze().cpu())
                all_targets.append(batch_y.cpu())
            
                total_correct += (torch.abs(preds - batch_y) < 0.05).sum().item()
                total_samples += batch_y.size(0)

        preds_flat = torch.cat(all_preds).numpy()
        targets_flat = torch.cat(all_targets).numpy()
    
        if epoch == 0 or (epoch+1) % 10 == 0:
            utils.plot_pred_vs_true(preds_flat, targets_flat, epoch, tag)
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        val_acc = total_correct / total_samples if total_samples > 0 else 0.0

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    utils.save_artifacts(EFCmodel, optimizer, train_loss_history, val_loss_history, lr_history, tag)

if __name__ == "__main__":
    main()