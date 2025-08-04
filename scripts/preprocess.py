import os
import numpy as np
from torchvision import datasets
from utils import ensure_dir

def to_npy(split, cifar):
    X = cifar.data  # (N,32,32,3) uint8
    y = np.array(cifar.targets, dtype=np.int64)
    out_dir = ensure_dir("data/processed/cifar10")
    np.save(os.path.join(out_dir, f"{split}_X.npy"), X)
    np.save(os.path.join(out_dir, f"{split}_y.npy"), y)
    print(f"Saved {split}: X={X.shape}, y={y.shape} to {out_dir}")

def main():
    root = ensure_dir("data/raw")
    train = datasets.CIFAR10(root=root, train=True, download=True)
    test  = datasets.CIFAR10(root=root, train=False, download=True)
    to_npy("train", train)
    to_npy("test", test)

if __name__ == "__main__":
    main()