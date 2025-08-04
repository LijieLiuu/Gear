import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms

from utils import ensure_dir, CIFAR_MEAN, CIFAR_STD

def list_images(root):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(r, f))
    return sorted(paths)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    softmax = nn.Softmax(dim=1)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    imgs = list_images(args.input_dir)
    recs = []
    with torch.no_grad():
        for p in tqdm(imgs, desc="Scoring synth"):
            img = Image.open(p).convert("RGB").resize((32,32))
            x = tfm(img).unsqueeze(0).to(device)
            logits = model(x)
            prob, cls = softmax(logits).max(dim=1)
            recs.append({"path": os.path.relpath(p, args.input_dir), "conf": float(prob.item()), "pred": int(cls.item())})

    df = pd.DataFrame(recs).sort_values("conf", ascending=False)
    if args.min_conf is not None:
        df = df[df["conf"] >= args.min_conf]
    if args.topk is not None and len(df) > args.topk:
        df = df.head(args.topk)

    out_dir = ensure_dir(args.output_dir)
    df.to_csv(os.path.join(out_dir, "scores.csv"), index=False)

    # 复制筛选后的图像
    from utils import copy_tree
    copy_tree(args.input_dir, out_dir, file_list=df["path"].tolist())
    print(f"Selected {len(df)} images to {out_dir}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="data/synth/50k")
    ap.add_argument("--output_dir", type=str, default="data/synth/top10k")
    ap.add_argument("--ckpt", type=str, default="checkpoints/baseline_resnet18.pth")
    ap.add_argument("--topk", type=int, default=10000)
    ap.add_argument("--min_conf", type=float, default=0.90)
    return ap.parse_args()

if __name__ == "__main__":
    main(parse_args())