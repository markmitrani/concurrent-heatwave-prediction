import os
import json
import math
from datetime import datetime
import matplotlib.pyplot as plt
import torch

def save_artifacts(model, optimizer, train_hist, val_hist, output_dir="outputs"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(output_dir, timestamp)
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

    # Save loss curve plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
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
