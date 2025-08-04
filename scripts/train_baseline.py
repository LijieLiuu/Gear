import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18
from sklearn.model_selection import StratifiedShuffleSplit

from datasets import NpyLabeledDataset, NpyTestDataset, get_train_transform_weak, get_weak_transform
from utils import ensure_dir, set_seed, save_json

def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def make_loaders(batch_size=64, seed=42):
    X = np.load("data/processed/cifar10/train_X.npy")
    y = np.load("data/processed/cifar10/train_y.npy")
    test = NpyTestDataset("data/processed/cifar10/test_X.npy",
                          "data/processed/cifar10/test_y.npy",
                          transform=get_weak_transform())

    # 80/20 Stratified split for train/val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(sss.split(X, y))

    train_ds = NpyLabeledDataset("data/processed/cifar10/train_X.npy",
                                 "data/processed/cifar10/train_y.npy",
                                 transform=get_train_transform_weak())
    val_ds   = NpyLabeledDataset("data/processed/cifar10/train_X.npy",
                                 "data/processed/cifar10/train_y.npy",
                                 transform=get_weak_transform())

    train_dl = DataLoader(Subset(train_ds, tr_idx), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(Subset(val_ds, val_idx),  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_dl, val_dl, test_dl

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, test_dl = make_loaders(args.batch_size, args.seed)

    model = resnet18(num_classes=10)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    curves = []
    best_val = -1.0
    best_path = ensure_dir("checkpoints")
    best_path = os.path.join(best_path, "baseline_resnet18.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_acc, n_tr = 0.0, 0.0, 0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            bs = yb.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += (logits.argmax(1) == yb).float().sum().item()
            n_tr    += bs
        tr_loss /= n_tr
        tr_acc  /= n_tr

        # val
        model.eval()
        va_loss, va_acc, n_va = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in tqdm(val_dl, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                bs = yb.size(0)
                va_loss += loss.item() * bs
                va_acc  += (logits.argmax(1) == yb).float().sum().item()
                n_va    += bs
        va_loss /= n_va
        va_acc  /= n_va



        te_acc, n_te = 0.0, 0
        with torch.no_grad():
            for xb, yb in tqdm(test_dl, desc=f"Epoch {epoch}/{args.epochs} [test]"):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                bs = yb.size(0)
                te_acc += (logits.argmax(1) == yb).float().sum().item()
                n_te   += bs
        te_acc /= n_te

        curves.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc":  tr_acc,
            "val_loss":   va_loss,
            "val_acc":    va_acc,
            "test_acc":   te_acc,
        })

        print(f"[Epoch {epoch}] train_acc={tr_acc:.4f} val_acc={va_acc:.4f} test_acc={te_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)


    df = pd.DataFrame(curves)
    out_dir = ensure_dir("results")
    csv_path = os.path.join(out_dir, "baseline_curves.csv")
    df.to_csv(csv_path, index=False)
    save_json({"best_val_acc": best_val, "best_ckpt": best_path}, os.path.join(out_dir, "baseline.json"))
    print(f"Saved curves to {csv_path} and best ckpt to {best_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    train(parse_args())