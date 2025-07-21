import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_and_save(X, y, prefix, out_dir, unlabeled_frac=0.2):
    os.makedirs(out_dir, exist_ok=True)
    X_lab, X_unlab, y_lab, _ = train_test_split(
        X, y, test_size=unlabeled_frac, stratify=y, random_state=42)
    np.save(os.path.join(out_dir, f"{prefix}_X_lab.npy"), X_lab)
    np.save(os.path.join(out_dir, f"{prefix}_y_lab.npy"), y_lab)
    np.save(os.path.join(out_dir, f"{prefix}_X_unlab.npy"), X_unlab)
    print(f"Split {prefix}:")
    print(f"  Labeled   -> {prefix}_X_lab.npy  ({X_lab.shape}), {prefix}_y_lab.npy")
    print(f"  Unlabeled -> {prefix}_X_unlab.npy  ({X_unlab.shape})")

if __name__ == "__main__":
    # CIFAR-10 第一个 batch 的示例，你也可以循环所有 batch
    X_cifar = np.load("data/processed/cifar10/cifar10_batch1_X.npy")
    y_cifar = np.load("data/processed/cifar10/cifar10_batch1_y.npy")
    split_and_save(X_cifar, y_cifar, "cifar10", "data/splits")

    # 20 Newsgroups 文本数据
    arr = np.load("data/processed/20news.npz", allow_pickle=True)
    docs = arr["docs"]
    labels = arr["labels"]
    split_and_save(docs, labels, "20news", "data/splits")