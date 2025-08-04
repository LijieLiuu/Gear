import os
import argparse
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18

from datasets import (
    NpyLabeledDataset, NpyUnlabeledDataset, NpyTestDataset,
    ImageFolderUnlabeled, get_train_transform_weak, get_train_transform_strong, get_weak_transform
)
from utils import ensure_dir, set_seed, save_json, linear_ramp

class AverageMeter:
    def __init__(self): self.v=0; self.s=0
    def add(self, v, n): self.v += v*n; self.s += n
    def mean(self): return self.v / max(self.s, 1)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    lab = NpyLabeledDataset("data/splits/X_lab.npy", "data/splits/y_lab.npy",
                            transform=get_train_transform_weak())
    unlab_real = NpyUnlabeledDataset("data/splits/X_unlab.npy",
                                     transform_w=get_train_transform_weak(),
                                     transform_s=get_train_transform_strong())
    if args.with_synth:
        unlab_synth = ImageFolderUnlabeled(args.synth_dir,
                                           transform_w=get_train_transform_weak(),
                                           transform_s=get_train_transform_strong())
        unlabeled = ConcatDataset([unlab_real, unlab_synth])
    else:
        unlabeled = unlab_real

    test = NpyTestDataset("data/processed/cifar10/test_X.npy",
                          "data/processed/cifar10/test_y.npy",
                          transform=get_weak_transform())

    lab_dl = DataLoader(lab, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    unlab_dl = DataLoader(unlabeled, batch_size=args.batch_size*2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = resnet18(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(reduction="none")  # 对无标签部分我们自己做 mask
    softmax = nn.Softmax(dim=1)

    total_steps = args.epochs * min(len(lab_dl), len(unlab_dl))
    tau_start, tau_end = args.tau_start, args.tau_end
    wu_start,  wu_end  = args.wu_start,  args.wu_end
    ramp_steps = int(total_steps * args.ramp_ratio)

    curves = []
    global_step = 0

    best_test = -1.0
    out_dir = ensure_dir("checkpoints")
    tag = "ssl_real" if not args.with_synth else "ssl_real_plus_synth"
    ckpt_path = os.path.join(out_dir, f"{tag}_resnet18.pth")

    for ep in range(1, args.epochs + 1):
        model.train()
        sup_meter, unsup_meter = AverageMeter(), AverageMeter()

        for (xb, yb), (uw, us) in tqdm(zip(lab_dl, unlab_dl), total=min(len(lab_dl), len(unlab_dl)), desc=f"Epoch {ep}/{args.epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            uw, us = uw.to(device), us.to(device)

            # ramp
            tau = linear_ramp(tau_start, tau_end, global_step, ramp_steps)
            wu  = linear_ramp(wu_start,  wu_end,  global_step, ramp_steps)

            opt.zero_grad()
            # supervised
            logits_sup = model(xb)
            loss_sup = ce(logits_sup, yb).mean()

            # unlabeled - pseudo labels from weak view
            with torch.no_grad():
                logits_w = model(uw)
                probs_w  = softmax(logits_w)
                confs, pseudo = probs_w.max(dim=1)
                mask = (confs >= tau).float()

            logits_s = model(us)
            loss_unsup_all = ce(logits_s, pseudo)  # CE with pseudo labels
            if mask.sum() > 0:
                loss_unsup = (loss_unsup_all * mask).sum() / mask.sum()
            else:
                loss_unsup = torch.tensor(0.0, device=device)

            loss = loss_sup + args.lambda_u * wu * loss_unsup
            loss.backward()
            opt.step()

            sup_meter.add(loss_sup.item(), xb.size(0))
            unsup_meter.add(loss_unsup.item(), us.size(0))
            global_step += 1

        # test eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item()
                total   += yb.size(0)
        test_acc = correct / total

        curves.append({
            "epoch": ep,
            "sup_loss": sup_meter.mean(),
            "unsup_loss": unsup_meter.mean(),
            "tau": float(tau),
            "wu": float(wu),
            "test_acc": test_acc,
        })
        print(f"[Epoch {ep}] sup={sup_meter.mean():.4f} unsup={unsup_meter.mean():.4f} tau={tau:.3f} wu={wu:.3f} test_acc={test_acc:.4f}")

        if test_acc > best_test:
            best_test = test_acc
            torch.save(model.state_dict(), ckpt_path)

    # save logs
    df = pd.DataFrame(curves)
    res_dir = ensure_dir("results")
    csv = os.path.join(res_dir, f"{tag}_curves.csv")
    df.to_csv(csv, index=False)
    save_json({"best_test_acc": best_test, "best_ckpt": ckpt_path}, os.path.join(res_dir, f"{tag}.json"))
    print(f"Saved {csv} and checkpoint {ckpt_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)  # 与报告统一
    ap.add_argument("--lambda_u", type=float, default=1.0)       # 一致性正则权重
    ap.add_argument("--tau_start", type=float, default=0.90)
    ap.add_argument("--tau_end", type=float, default=0.95)
    ap.add_argument("--wu_start", type=float, default=0.10)
    ap.add_argument("--wu_end", type=float, default=1.00)
    ap.add_argument("--ramp_ratio", type=float, default=0.3)     # 前 30% steps ramp
    ap.add_argument("--with_synth", action="store_true")
    ap.add_argument("--synth_dir", type=str, default="data/synth/50k")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

if __name__ == "__main__":
    train(parse_args())