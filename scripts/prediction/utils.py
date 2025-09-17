import os
import json
import math
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_mse_loss(y_pred, y_true, mask_threshold=0.05):
    mask = (y_true >= mask_threshold)
    y_pred_masked = y_pred[mask]
    y_true_masked = y_true[mask]
    return F.mse_loss(y_pred_masked, y_true_masked, reduction='mean')

class MaskedMSELoss(nn.Module):
    def __init__(self, threshold=0.05):
        super().__init__()
        self.threshold = threshold

    def forward(self, input, target):
        # Create mask where target >= threshold
        mask = (target >= self.threshold).float()
        
        # Compute squared error
        loss = (input - target) ** 2
        
        # Apply mask and compute mean only over valid elements
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss

def save_artifacts(model, optimizer, train_hist, val_hist, lr_hist, tag, output_dir="outputs"):
    out_dir = os.path.join(output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Save model checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(out_dir, "checkpoint.pt"))

    # Save training/validation loss history
    with open(os.path.join(out_dir, "train_loss.json"), "w") as f:
        json.dump(train_hist, f)
    with open(os.path.join(out_dir, "val_loss.json"), "w") as f:
        json.dump(val_hist, f)

    # Better styling
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['figure.figsize'] = (8,5)
    plt.rcParams['figure.dpi'] = 600

    # Save loss curve plot
    # plt.figure(figsize=(8, 5))
    plt.plot(train_hist, color='teal', label='Train Loss')
    plt.plot(val_hist, color='sandybrown', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.plot(lr_hist, color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate over Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lr_curve.png"))
    plt.close()

def plot_pred_vs_true(pred_list, true_list, epoch, tag, output_dir = 'outputs'):
    out_dir = os.path.join(output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Better styling
    plt.style.use("ggplot")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams['figure.figsize'] = (16,9)
    plt.rcParams['figure.dpi'] = 600

    plt.figure()
    plt.scatter(true_list, pred_list, alpha=0.5, color='darkturquoise')
    plt.xlabel("True Participation Probability")
    plt.ylabel("Predicted Probability")
    plt.title(f"Predicted vs True (Epoch {epoch+1})")
    plt.tight_layout()
    plt.grid(True)

    plt.savefig(f"{out_dir}/pred_vs_true_epoch_{epoch+1}.png")
    plt.close()

def get_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step, warmup_steps=warmup_steps, total_steps=total_steps):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

# TODO remove the functions below, not used

def compute_accuracy(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
    return total_correct / total_samples if total_samples > 0 else 0.0

def training_step(model, batch_x, batch_y, loss_fn):
    model.train()
    logits = model(batch_x)
    loss = loss_fn(logits, batch_y)
    return loss

def validation_step(model, batch_x, batch_y, loss_fn):
    model.eval()
    with torch.no_grad():
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch_y).float().mean()
    return loss.item(), acc.item()
